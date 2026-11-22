"""Build two assignments of the fixed E32 training pool."""

from collections import Counter, defaultdict
import json
import random

from exp_manager.utils.project_root_path import PROJECT_ROOT_PATH
from fedsec.validation import positive_int


E32_DATA_PROTOCOL = "e32"
E32_MANIFEST = "configs/cifar10_45k_5k.json"


def load_e32_manifest() -> dict:
    manifest = json.loads((PROJECT_ROOT_PATH / E32_MANIFEST).read_text(encoding="utf-8"))
    manifest["final_test_ranges"] = [
        {"file": "cifar-10-batches-py/test_batch", "start": 0, "end": 10000}
    ]
    manifest["scope"] = "full"
    return manifest


def build_paired_assignments(manifest: dict, seed: int, loader):
    """Validate file/index identities, then assign every training image in both cases."""
    source = (PROJECT_ROOT_PATH / manifest["source_dir"]).resolve()
    owners = sorted(item["client_id"] for item in manifest["clients"])
    if not owners or owners != list(range(len(owners))):
        raise ValueError("Training manifest must identify clients 0 through N-1")
    original_ranges = [segment for item in manifest["clients"] for segment in item["ranges"]]
    occupied = set()
    labels = {}

    def read_ranges(ranges):
        normalized, samples = [], []
        for segment in ranges:
            path = (source / segment["file"]).resolve()
            if not path.is_relative_to(source):
                raise ValueError("Dataset file must stay inside source_dir")
            file = path.relative_to(source).as_posix()
            if file not in labels:
                labels[file] = loader.labels(path)
                if len(labels[file]) != loader.record_count(path):
                    raise ValueError(f"Pixel and label counts differ in {file}")
            start = positive_int(segment["start"], "range start", 0)
            end = positive_int(segment["end"], "range end")
            if not start < end <= len(labels[file]):
                raise ValueError(f"Dataset range is empty or out of bounds in {file}")
            records = [(file, index) for index in range(start, end)]
            if occupied.intersection(records):
                raise ValueError(f"Training/validation/final test indices overlap in {file}")
            occupied.update(records)
            samples.extend(records)
            normalized.append({"file": file, "start": start, "end": end})
        if not samples:
            raise ValueError("Training, validation, and final test must all be nonempty")
        return normalized, samples

    training_ranges, training = read_ranges(original_ranges)
    validation_ranges, validation = read_ranges(manifest["test_ranges"])
    final_test_ranges, final_test = read_ranges(manifest["final_test_ranges"])
    counts = (len(training), len(validation), len(final_test))
    if manifest["scope"] == "full" and (len(owners), counts) != (10, (45000, 5000, 10000)):
        raise ValueError("E32 requires 10 clients and 45000/5000/10000 train/validation/test images")
    per_client, extra = divmod(len(training), len(owners))
    if extra or per_client == 0:
        raise ValueError("The training pool must divide equally across clients")

    by_class = [[] for _ in range(10)]
    for file, index in sorted(training):
        by_class[labels[file][index]].append((file, index))
    rng = random.Random(positive_int(seed, "seed", 0))
    for samples in by_class:
        rng.shuffle(samples)
    class_order = list(range(10))
    rng.shuffle(class_order)
    class_sorted = [item for label in class_order for item in by_class[label]]
    conditions = {
        "iid": [class_sorted[client::len(owners)] for client in owners],
        "non_iid": [class_sorted[client * per_client:(client + 1) * per_client] for client in owners],
    }
    training_set = set(training)
    result = {}
    for distribution, assigned in conditions.items():
        if any(len(client) != per_client for client in assigned) or {
            record for client in assigned for record in client
        } != training_set:
            raise ValueError(f"{distribution} does not preserve the E32 training pool and counts")
        clients, class_counts = [], []
        for client_id, records in enumerate(assigned):
            indices = defaultdict(list)
            classes = Counter()
            for file, index in sorted(records):
                indices[file].append(index)
                classes[labels[file][index]] += 1
            clients.append({
                "client_id": client_id,
                "indices_by_file": [{"file": file, "indices": values} for file, values in indices.items()],
            })
            class_counts.append({
                "client_id": client_id, "num_samples": len(records),
                "class_counts": {str(label): classes[label] for label in range(10)},
            })
        result[distribution] = (clients, class_counts)

    return {
        "source_dir": manifest["source_dir"], "scope": manifest["scope"],
        "training_ranges": training_ranges, "validation_ranges": validation_ranges,
        "final_test_ranges": final_test_ranges, "files": sorted(labels),
        "conditions": result,
    }
