"""매니저가 배정한 CIFAR-10 피클 파일의 범위를 정확히 읽는다."""

from functools import lru_cache
import hashlib
import json
from pathlib import Path
import pickle

import numpy as np
import torch
from torch.utils.data import Dataset

from fedsec.model import PREPROCESSING
from fedsec.validation import positive_int

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@lru_cache(maxsize=2)
def _read_batch(path: str, size: int, modified: int):
    del size, modified
    raw = Path(path).read_bytes()
    batch = pickle.loads(raw, encoding="bytes")
    data = np.asarray(batch.get(b"data", batch.get("data")))
    labels = np.asarray(batch.get(b"labels", batch.get("labels")))
    if data.ndim != 2 or data.shape[1] != 3072 or labels.shape != (len(data),):
        raise ValueError("Expected CIFAR-10 data [N,3072] and labels [N]")
    if data.dtype != np.uint8 or not np.issubdtype(labels.dtype, np.integer) or np.any((labels < 0) | (labels > 9)):
        raise ValueError("Invalid CIFAR-10 pixels or labels")
    return data, labels, hashlib.sha256(raw).hexdigest()


class CifarRangeDataset(Dataset):
    def __init__(self, payload: dict, limit: int | None = None):
        if limit is not None:
            positive_int(limit, "sample limit")
        if (
            not isinstance(payload, dict)
            or not payload.get("source_dir")
            or not (payload.get("ranges") or payload.get("indices_by_file"))
        ):
            raise ValueError("Distribute CIFAR-10 data before starting the experiment")
        root = Path(payload["source_dir"])
        root = (root if root.is_absolute() else PROJECT_ROOT / root).resolve()
        chunks, targets, records, seen = [], [], [], {}
        remaining = limit
        segments = [
            {**segment, "selection": "range"} for segment in payload.get("ranges", [])
        ] + [
            {**segment, "selection": "indices"}
            for segment in payload.get("indices_by_file", [])
        ]
        for segment in segments:
            path = (root / segment["file"]).resolve()
            if not path.is_relative_to(root):
                raise ValueError("Dataset file must stay inside source_dir")
            stat = path.stat()
            data, labels, digest = _read_batch(str(path), stat.st_size, stat.st_mtime_ns)
            if segment["selection"] == "range":
                start = positive_int(segment["start"], "range start", 0)
                end = positive_int(segment["end"], "range end", 0)
                if not 0 <= start < end <= len(data):
                    raise ValueError("Dataset range is empty or out of bounds")
                for previous in seen.setdefault(str(path), []):
                    if isinstance(previous, set):
                        if any(start <= index < end for index in previous):
                            raise ValueError("Dataset ranges overlap")
                    elif start < previous[1] and previous[0] < end:
                        raise ValueError("Dataset ranges overlap")
                seen[str(path)].append((start, end))
                sample_indices = np.arange(start, end, dtype=np.int64)
            else:
                raw_indices = segment.get("indices")
                if not isinstance(raw_indices, list) or not raw_indices:
                    raise ValueError("Dataset indices must be a non-empty list")
                if any(
                    type(index) is not int or not 0 <= index < len(data)
                    for index in raw_indices
                ) or len(set(raw_indices)) != len(raw_indices):
                    raise ValueError("Dataset indices are duplicated or out of bounds")
                previous_assignments = seen.setdefault(str(path), [])
                if any(
                    isinstance(previous, tuple)
                    and any(previous[0] <= index < previous[1] for index in raw_indices)
                    for previous in previous_assignments
                ):
                    raise ValueError("Dataset indices overlap")
                assigned = {index for previous in previous_assignments if isinstance(previous, set) for index in previous}
                if assigned.intersection(raw_indices):
                    raise ValueError("Dataset indices overlap")
                previous_assignments.append(set(raw_indices))
                sample_indices = np.asarray(raw_indices, dtype=np.int64)
            if remaining is not None:
                sample_indices = sample_indices[:remaining]
            if len(sample_indices):
                chunks.append(data[sample_indices])
                targets.append(labels[sample_indices])
                record = {
                    "file": path.relative_to(root).as_posix(),
                    "sha256": digest,
                }
                if segment["selection"] == "range":
                    record.update(
                        start=int(sample_indices[0]),
                        end=int(sample_indices[-1]) + 1,
                    )
                else:
                    record["indices"] = sample_indices.tolist()
                records.append(record)
                if remaining is not None:
                    remaining -= len(sample_indices)
        if not chunks:
            raise ValueError("Dataset is empty")
        self.images = np.concatenate(chunks).reshape(-1, 3, 32, 32)
        self.labels = np.concatenate(targets).astype(np.int64)
        self.fingerprint = hashlib.sha256(json.dumps(records, sort_keys=True).encode()).hexdigest()
        self.mean = torch.tensor(PREPROCESSING["mean"]).reshape(3, 1, 1)
        self.std = torch.tensor(PREPROCESSING["std"]).reshape(3, 1, 1)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        image = torch.from_numpy(self.images[index].copy()).float().div_(255)
        return (image - self.mean) / self.std, int(self.labels[index])


def load_datasets(manifest: dict, settings) -> tuple[dict[int, Dataset], Dataset]:
    """이미 결정된 데이터 배정표를 읽는다. 분할 정책/다운로드/네트워크 처리는 하지 않는다.

    manifest: source_dir, clients=[{client_id, ranges}], test_ranges=[{file,start,end}].
    학습·테스트 및 클라이언트끼리 겹치는 범위를 거부한다.
    """
    source = manifest["source_dir"]
    clients = manifest["clients"]
    if not clients:
        raise ValueError("Manifest contains no clients")
    ids = [positive_int(item["client_id"], "client_id", 0) for item in clients]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate client ID in manifest")
    root = Path(source)
    root = (root if root.is_absolute() else PROJECT_ROOT / root).resolve()
    seen_ranges = {}
    seen_indices = {}
    selections = [
        [
            {**segment, "selection": "range"}
            for segment in item.get("ranges", [])
        ]
        + [
            {**segment, "selection": "indices"}
            for segment in item.get("indices_by_file", [])
        ]
        for item in clients
    ] + [[{**segment, "selection": "range"} for segment in manifest["test_ranges"]]]
    for selection in selections:
        for segment in selection:
            path = (root / segment["file"]).resolve()
            if not path.is_relative_to(root):
                raise ValueError("Dataset file must stay inside source_dir")
            if segment["selection"] == "range":
                start = positive_int(segment["start"], "range start", 0)
                end = positive_int(segment["end"], "range end")
                if start >= end:
                    raise ValueError("Empty dataset range")
                if any(
                    start < old_end and old_start < end
                    for old_start, old_end in seen_ranges.get(path, [])
                ) or any(
                    start <= index < end for index in seen_indices.get(path, set())
                ):
                    raise ValueError("Client/test dataset ranges overlap")
                seen_ranges.setdefault(path, []).append((start, end))
            else:
                indices = segment.get("indices")
                if not isinstance(indices, list) or any(
                    type(index) is not int or index < 0 for index in indices
                ) or len(indices) != len(set(indices)):
                    raise ValueError("Invalid dataset indices")
                selected = seen_indices.setdefault(path, set())
                if selected.intersection(indices) or any(
                    start <= index < end for start, end in seen_ranges.get(path, [])
                ):
                    raise ValueError("Client/test dataset indices overlap")
                selected.update(indices)
    datasets = {item["client_id"]: CifarRangeDataset({"source_dir": source, **{
                                                        key: item.get(key, [])
                                                        for key in ("ranges", "indices_by_file")
                                                    }},
                                                    settings.max_train_samples) for item in clients}
    test = CifarRangeDataset({"source_dir": source, "ranges": manifest["test_ranges"]}, settings.max_test_samples)
    return datasets, test
