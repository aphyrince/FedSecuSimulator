import threading
import hashlib
import json
from pathlib import Path

from exp_manager.services.data_loader import CifarDataLoader
from exp_manager.services.data_splitter import DataSplitter
from exp_manager.services.e32_data import E32_DATA_PROTOCOL, E32_MANIFEST, build_paired_assignments, load_e32_manifest
from exp_manager.services.rest_client import RestClient
from exp_manager.services.sequential_clients import SequentialClientRunner
from exp_manager.utils.data_config import read_data_config
from exp_manager.utils.project_root_path import PROJECT_ROOT_PATH
from fedsec.data import dataset_fingerprint


class DataDistributionManager:
    """데이터 설정을 읽고, 테스트·훈련 범위를 각 서버에 분배한다."""

    def __init__(self, process_manager, client_count: int, *, e32_train_samples_per_client: int | None = None):
        self.process_manager = process_manager
        self.client_count = client_count
        self.e32_train_samples_per_client = e32_train_samples_per_client
        self.loader = CifarDataLoader()
        self.splitter = DataSplitter()
        self.rest = RestClient()
        self.lock = threading.RLock()
        self.file_hash_cache = {}
        self.state = {
            "configured": False,
            "message": "데이터가 분배되지 않았습니다.",
            "seed": None,
            "split_id": None,
            "split_manifest_path": None,
            "distribution": None,
            "client_class_counts": [],
            "client_sample_counts": [],
            "training_image_set": None,
            "distribution_summaries": {},
        }

    def status(self) -> dict:
        with self.lock:
            return {
                "configured": self.state["configured"],
                "message": self.state["message"],
                "config": self.state.get("data_config") or read_data_config(),
                "data_protocol": self.state.get("data_protocol"),
                "scope": self.state.get("scope"),
                "validation_range": self.state.get("validation_range"),
                "final_test_range": self.state.get("final_test_range"),
                "evaluation_sets": self.state.get("evaluation_sets"),
                "test_range": self.state.get("test_range"),
                "client_ranges": self.state.get("client_ranges", []),
                "seed": self.state.get("seed"),
                "split_id": self.state.get("split_id"),
                "split_manifest_path": self.state.get("split_manifest_path"),
                "distribution": self.state.get("distribution"),
                "client_class_counts": self.state.get("client_class_counts", []),
                "client_sample_counts": self.state.get("client_sample_counts", []),
                "training_image_set": self.state.get("training_image_set"),
                "distribution_summaries": self.state.get("distribution_summaries", {}),
            }

    def distribute(self, seed: int = 42, distribution: str = "iid", *, data_protocol=None) -> dict:
        with self.lock:
            if distribution not in {"iid", "non_iid"}:
                raise ValueError("데이터 분포는 iid 또는 non_iid여야 합니다.")
            if data_protocol == E32_DATA_PROTOCOL:
                return self._distribute_e32(seed, distribution)
            if data_protocol is not None:
                raise ValueError("지원하지 않는 데이터 프로토콜입니다.")
            config = read_data_config()
            source_path = Path(config["source_path"])
            if not source_path.is_dir():
                raise FileNotFoundError(
                    f"SOURCE_DIR를 찾을 수 없습니다: {source_path}"
                )

            train_files = self.loader.resolve_files(
                source_path, config["train_files"], "train"
            )
            test_files = self.loader.resolve_files(
                source_path, config["test_files"], "test"
            )
            train_segments = self._segments(train_files, source_path)
            test_segments = self._segments(test_files, source_path)
            self._validate_segments(train_segments, test_segments)

            labels_by_file = {
                segment["file"]: self.loader.labels(source_path / segment["file"])
                for segment in train_segments
            }
            client_assignments, _ = self.splitter.split(
                train_segments,
                self.client_count,
                seed,
                distribution=distribution,
                labels_by_file=labels_by_file,
            )
            client_class_counts = self._count_classes(
                client_assignments, labels_by_file
            )
            client_sample_counts = [
                {
                    "client_id": item["client_id"],
                    "num_samples": item["num_samples"],
                }
                for item in client_class_counts
            ]
            test_payload = {
                "source_dir": config["source_dir"],
                "ranges": test_segments,
            }
            self._send_test_range(test_payload)
            sent_client_ranges = self._send_client_ranges(
                config["source_dir"], client_assignments
            )
            training_image_set = self._training_image_set(sent_client_ranges)
            file_hashes = {
                str(path.relative_to(source_path)).replace("\\", "/"): self._file_hash(
                    path
                )
                for path in [*train_files, *test_files]
            }
            distribution_summaries = {}
            for summary_distribution in ("iid", "non_iid"):
                if summary_distribution == distribution:
                    summary_assignments = client_assignments
                    summary_class_counts = client_class_counts
                    summary_image_set = training_image_set
                else:
                    summary_assignments, _ = self.splitter.split(
                        train_segments,
                        self.client_count,
                        seed,
                        distribution=summary_distribution,
                        labels_by_file=labels_by_file,
                    )
                    summary_class_counts = self._count_classes(
                        summary_assignments, labels_by_file
                    )
                    summary_image_set = self._training_image_set(
                        [
                            {"client_id": client_id, **assignment}
                            for client_id, assignment in enumerate(summary_assignments)
                        ]
                    )
                summary_sample_counts = [
                    {
                        "client_id": item["client_id"],
                        "num_samples": item["num_samples"],
                    }
                    for item in summary_class_counts
                ]
                distribution_summaries[summary_distribution] = {
                    "distribution": summary_distribution,
                    "client_class_counts": summary_class_counts,
                    "client_sample_counts": summary_sample_counts,
                    "training_image_set": {
                        "sample_count": summary_image_set["sample_count"],
                        "sha256": summary_image_set["sha256"],
                    },
                    "comparison_protocol": self._comparison_protocol(
                        summary_distribution,
                        summary_sample_counts,
                        summary_class_counts,
                        summary_image_set,
                    ),
                }
            split_payload = {
                "seed": seed,
                "source_dir": config["source_dir"],
                "distribution": distribution,
                "file_sha256": file_hashes,
                "test_range": test_payload,
                "client_ranges": sent_client_ranges,
                "client_class_counts": client_class_counts,
                "client_sample_counts": client_sample_counts,
                "training_image_set": training_image_set,
                "comparison_protocol": self._comparison_protocol(
                    distribution,
                    client_sample_counts,
                    client_class_counts,
                    training_image_set,
                ),
            }
            split_id, split_manifest_path = self.save_split_manifest(split_payload)

            self.state.update(
                configured=True,
                message="훈련 및 테스트 데이터 범위를 분배했습니다.",
                test_range=test_payload,
                client_ranges=sent_client_ranges,
                seed=seed,
                split_id=split_id,
                split_manifest_path=str(split_manifest_path),
                distribution=distribution,
                client_class_counts=client_class_counts,
                client_sample_counts=client_sample_counts,
                training_image_set=training_image_set,
                distribution_summaries=distribution_summaries,
                data_protocol=None, scope=None, data_config=None,
                validation_range=None, final_test_range=None, evaluation_sets=None,
            )
            return self.status()

    @staticmethod
    def save_split_manifest(payload: dict, directory: Path | None = None) -> tuple[str, Path]:
        split_id = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]
        directory = directory or PROJECT_ROOT_PATH / "results" / "data_splits"
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{split_id}.json"
        path.write_text(json.dumps({"split_id": split_id, **payload}, ensure_ascii=False, indent=2), encoding="utf-8")
        return split_id, path

    def prepare_e32_splits(self, seed: int) -> dict:
        """Build and verify both E32 splits without making any Flask requests."""
        paired = build_paired_assignments(
            load_e32_manifest(), seed, self.loader,
            training_samples_per_client=self.e32_train_samples_per_client,
        )
        source = paired["source_dir"]
        root = PROJECT_ROOT_PATH / source
        file_hashes = {file: self._file_hash(root / file) for file in paired["files"]}
        validation = {"source_dir": source, "ranges": paired["validation_ranges"], "split": "validation"}
        final_test = {"source_dir": source, "ranges": paired["final_test_ranges"], "split": "test"}
        evaluation_sets = {}
        for selection in (validation, final_test):
            ranges = selection["ranges"]
            evaluation_sets[selection["split"]] = {
                "sample_count": sum(item["end"] - item["start"] for item in ranges),
                "dataset_fingerprint": dataset_fingerprint([
                    {**item, "sha256": file_hashes[item["file"]]} for item in ranges
                ]),
            }
        return {
            distribution: {
                "data_protocol": E32_DATA_PROTOCOL, "scope": paired["scope"],
                "reference_manifest": E32_MANIFEST, "seed": seed,
                "source_dir": source, "distribution": distribution, "file_sha256": file_hashes,
                "test_range": validation, "validation_range": validation, "final_test_range": final_test,
                "evaluation_sets": evaluation_sets, "client_ranges": clients,
                "client_class_counts": counts,
                "client_sample_counts": [{"client_id": row["client_id"], "num_samples": row["num_samples"]} for row in counts],
                "training_image_set": self._training_image_set(clients),
            }
            for distribution, (clients, counts) in paired["conditions"].items()
        }

    def _distribute_e32(self, seed: int, distribution: str) -> dict:
        splits = self.prepare_e32_splits(seed)
        split = splits[distribution]
        if len(split["client_ranges"]) != self.client_count:
            raise ValueError(f"E32 requires {len(split['client_ranges'])} clients for this dataset")
        paths = {name: self.save_split_manifest(payload) for name, payload in splits.items()}
        self.state["configured"] = False
        self._send_test_range({**split["validation_range"], "final_test_range": split["final_test_range"]})
        self._send_client_ranges(split["source_dir"], [
            {"indices_by_file": client["indices_by_file"]} for client in split["client_ranges"]
        ])
        split_id, path = paths[distribution]
        summaries = {}
        for name, payload in splits.items():
            summaries[name] = {
                key: payload[key] for key in ("distribution", "client_class_counts", "client_sample_counts")
            } | {
                "training_image_set": {key: payload["training_image_set"][key] for key in ("sample_count", "sha256")},
                "split_manifest_path": str(paths[name][1]),
            }
        self.state.update(
            split, configured=True, split_id=split_id, split_manifest_path=str(path),
            distribution_summaries=summaries,
            message="E32 학습·검증·공식 테스트 데이터를 분배했습니다.",
            data_config={
                "source_dir": split["source_dir"],
                "train_files": [item["file"] for item in split["training_image_set"]["files"]],
                "test_files": [item["file"] for item in split["final_test_range"]["ranges"]],
            },
        )
        return self.status()

    @staticmethod
    def _training_image_set(client_assignments: list[dict]) -> dict:
        indices_by_file = {}
        for assignment in client_assignments:
            for segment in assignment.get("ranges", []):
                indices_by_file.setdefault(segment["file"], set()).update(
                    range(segment["start"], segment["end"])
                )
            for segment in assignment.get("indices_by_file", []):
                indices_by_file.setdefault(segment["file"], set()).update(
                    segment["indices"]
                )
        files = [
            {"file": file_name, "indices": sorted(indices)}
            for file_name, indices in sorted(indices_by_file.items())
        ]
        canonical = json.dumps(files, sort_keys=True, separators=(",", ":"))
        return {
            "sample_count": sum(len(item["indices"]) for item in files),
            "sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
            "files": files,
        }

    @staticmethod
    def _comparison_protocol(
        distribution: str,
        client_sample_counts: list[dict],
        client_class_counts: list[dict],
        training_image_set: dict,
    ) -> dict | None:
        if (
            len(client_sample_counts) != DataSplitter.COMPARISON_CLIENT_COUNT
            or training_image_set["sample_count"] != 5000
            or any(item["num_samples"] != 500 for item in client_sample_counts)
        ):
            return None
        if distribution == "iid":
            expected_classes = {str(label): 50 for label in range(10)}
            if any(
                item["class_counts"] != expected_classes
                for item in client_class_counts
            ):
                return None
        elif distribution == "non_iid":
            if any(
                sorted(item for item in counts["class_counts"].values() if item)
                != [250, 250]
                for counts in client_class_counts
            ):
                return None
            totals = {
                str(label): sum(
                    item["class_counts"][str(label)]
                    for item in client_class_counts
                )
                for label in range(10)
            }
            if any(total != 500 for total in totals.values()):
                return None
        else:
            return None
        return {
            "name": "cifar10_paired_5000",
            "version": 1,
            "pool_seed": DataSplitter.COMPARISON_POOL_SEED,
            "samples_per_client": 500,
            "samples_per_class": 500,
        }

    def _file_hash(self, path: Path) -> str:
        stat = path.stat()
        key = (str(path.resolve()), stat.st_size, stat.st_mtime_ns)
        if key not in self.file_hash_cache:
            self.file_hash_cache[key] = hashlib.sha256(path.read_bytes()).hexdigest()
        return self.file_hash_cache[key]

    @staticmethod
    def _count_classes(
        assignments: list[dict],
        labels_by_file: dict[str, list[int]],
    ) -> list[dict]:
        from collections import Counter

        counts = []
        for client_id, assignment in enumerate(assignments):
            classes = Counter()
            for segment in assignment.get("ranges", []):
                classes.update(
                    labels_by_file[segment["file"]][segment["start"] : segment["end"]]
                )
            for segment in assignment.get("indices_by_file", []):
                classes.update(
                    labels_by_file[segment["file"]][index]
                    for index in segment["indices"]
                )
            class_counts = {str(label): classes[label] for label in range(10)}
            counts.append(
                {
                    "client_id": client_id,
                    "num_samples": sum(class_counts.values()),
                    "class_counts": class_counts,
                }
            )
        return counts

    def _segments(self, files: list[Path], source_path: Path) -> list[dict]:
        return [
            {
                "file": str(path.relative_to(source_path)).replace("\\", "/"),
                "start": 0,
                "end": self.loader.record_count(path),
            }
            for path in files
        ]

    def _validate_segments(
        self, train_segments: list[dict], test_segments: list[dict]
    ):
        if sum(item["end"] for item in train_segments) == 0:
            raise ValueError("훈련 데이터가 비어 있습니다.")
        if sum(item["end"] for item in test_segments) == 0:
            raise ValueError("테스트 데이터가 비어 있습니다.")
        for train in train_segments:
            for test in test_segments:
                if train["file"] == test["file"] and train["start"] < test["end"] and test["start"] < train["end"]:
                    raise ValueError("훈련과 테스트 데이터 범위가 겹칩니다.")

    def _send_test_range(self, payload: dict):
        server_url = (
            f"http://{self.process_manager.host}:"
            f"{self.process_manager.server_port}"
        )
        self.rest.post_json(server_url + "/api/dataset/range", payload)

    def _send_client_ranges(self, source_dir: str, assignments: list[dict]):
        runner = getattr(self.process_manager, "sequential_runner", None)
        if isinstance(runner, SequentialClientRunner):
            runner.configure_datasets(source_dir, assignments)
            return [{"client_id": client_id, **assignment} for client_id, assignment in enumerate(assignments)]
        sent_ranges = []
        for client_id, assignment in enumerate(assignments):
            port = self.process_manager.client_base_port + client_id
            client_url = f"http://{self.process_manager.host}:{port}"
            payload = {"source_dir": source_dir, **assignment}
            self.rest.post_json(client_url + "/api/dataset/range", payload)
            sent_ranges.append({"client_id": client_id, **assignment})
        return sent_ranges
