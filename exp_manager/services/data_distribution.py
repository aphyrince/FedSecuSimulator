import threading
from pathlib import Path

from exp_manager.services.data_loader import CifarDataLoader
from exp_manager.services.data_splitter import DataSplitter
from exp_manager.services.rest_client import RestClient
from exp_manager.utils.data_config import read_data_config


class DataDistributionManager:
    """데이터 설정을 읽고, 테스트·훈련 범위를 각 서버에 분배한다."""

    def __init__(self, process_manager, client_count: int):
        self.process_manager = process_manager
        self.client_count = client_count
        self.loader = CifarDataLoader()
        self.splitter = DataSplitter()
        self.rest = RestClient()
        self.lock = threading.RLock()
        self.state = {
            "configured": False,
            "message": "데이터가 분배되지 않았습니다.",
        }

    def status(self) -> dict:
        with self.lock:
            return {
                "configured": self.state["configured"],
                "message": self.state["message"],
                "config": read_data_config(),
                "test_range": self.state.get("test_range"),
                "client_ranges": self.state.get("client_ranges", []),
            }

    def distribute(self, seed: int = 42) -> dict:
        with self.lock:
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

            client_ranges = self.splitter.split(
                train_segments, self.client_count, seed
            )
            test_payload = {
                "source_dir": config["source_dir"],
                "ranges": test_segments,
            }
            self._send_test_range(test_payload)
            sent_client_ranges = self._send_client_ranges(
                config["source_dir"], client_ranges
            )

            self.state.update(
                configured=True,
                message="훈련 및 테스트 데이터 범위를 분배했습니다.",
                test_range=test_payload,
                client_ranges=sent_client_ranges,
            )
            return self.status()

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

    def _send_client_ranges(self, source_dir: str, client_ranges: list[list[dict]]):
        sent_ranges = []
        for client_id, ranges in enumerate(client_ranges):
            port = self.process_manager.client_base_port + client_id
            client_url = f"http://{self.process_manager.host}:{port}"
            payload = {"source_dir": source_dir, "ranges": ranges}
            self.rest.post_json(client_url + "/api/dataset/range", payload)
            sent_ranges.append({"client_id": client_id, "ranges": ranges})
        return sent_ranges
