import threading
import hashlib
import json
from pathlib import Path

from exp_manager.services.data_loader import CifarDataLoader
from exp_manager.services.data_splitter import DataSplitter
from exp_manager.services.rest_client import RestClient
from exp_manager.utils.data_config import read_data_config
from exp_manager.utils.project_root_path import PROJECT_ROOT_PATH


class DataDistributionManager:
    """데이터 설정을 읽고, 테스트·훈련 범위를 각 서버에 분배한다."""

    def __init__(self, process_manager, client_count: int):
        self.process_manager = process_manager
        self.client_count = client_count
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
        }

    def status(self) -> dict:
        with self.lock:
            return {
                "configured": self.state["configured"],
                "message": self.state["message"],
                "config": read_data_config(),
                "test_range": self.state.get("test_range"),
                "client_ranges": self.state.get("client_ranges", []),
                "seed": self.state.get("seed"),
                "split_id": self.state.get("split_id"),
                "split_manifest_path": self.state.get("split_manifest_path"),
                "distribution": self.state.get("distribution"),
                "client_class_counts": self.state.get("client_class_counts", []),
            }

    def distribute(self, seed: int = 42, distribution: str = "iid") -> dict:
        with self.lock:
            if distribution not in {"iid", "non_iid"}:
                raise ValueError("데이터 분포는 iid 또는 non_iid여야 합니다.")
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
            test_payload = {
                "source_dir": config["source_dir"],
                "ranges": test_segments,
            }
            self._send_test_range(test_payload)
            sent_client_ranges = self._send_client_ranges(
                config["source_dir"], client_assignments
            )
            file_hashes = {
                str(path.relative_to(source_path)).replace("\\", "/"): self._file_hash(
                    path
                )
                for path in [*train_files, *test_files]
            }
            split_payload = {
                "seed": seed,
                "source_dir": config["source_dir"],
                "distribution": distribution,
                "file_sha256": file_hashes,
                "test_range": test_payload,
                "client_ranges": sent_client_ranges,
                "client_class_counts": client_class_counts,
            }
            split_id = hashlib.sha256(
                json.dumps(split_payload, sort_keys=True).encode("utf-8")
            ).hexdigest()[:16]
            split_dir = PROJECT_ROOT_PATH / "results" / "data_splits"
            split_dir.mkdir(parents=True, exist_ok=True)
            split_manifest_path = split_dir / f"{split_id}.json"
            split_manifest_path.write_text(
                json.dumps(
                    {"split_id": split_id, **split_payload},
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

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
            )
            return self.status()

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
        sent_ranges = []
        for client_id, assignment in enumerate(assignments):
            port = self.process_manager.client_base_port + client_id
            client_url = f"http://{self.process_manager.host}:{port}"
            payload = {"source_dir": source_dir, **assignment}
            self.rest.post_json(client_url + "/api/dataset/range", payload)
            sent_ranges.append({"client_id": client_id, **assignment})
        return sent_ranges
