import pickle
from pathlib import Path


class CifarDataLoader:
    """CIFAR 배치 파일의 위치와 레코드 수를 검사한다."""

    def resolve_files(self, source_path: Path, names: list[str], kind: str) -> list[Path]:
        cifar_dir = source_path / "cifar-10-batches-py"
        if names:
            files = [source_path / name for name in names]
        elif kind == "train" and cifar_dir.is_dir():
            files = [cifar_dir / f"data_batch_{batch_id}" for batch_id in range(1, 6)]
        elif kind == "test" and (cifar_dir / "test_batch").is_file():
            files = [cifar_dir / "test_batch"]
        elif kind == "train":
            files = sorted(source_path.glob("data_batch_*"))
        elif kind == "test":
            files = sorted([source_path / "test_batch"])
        else:
            files = []

        missing = [str(path) for path in files if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                "데이터 파일을 찾을 수 없습니다: " + ", ".join(missing)
            )
        if not files:
            raise FileNotFoundError(f"{kind} 데이터를 찾을 수 없습니다.")
        return files

    def record_count(self, file_path: Path) -> int:
        """CIFAR pickle의 data 항목 길이를 반환한다."""
        with file_path.open("rb") as source:
            batch = pickle.load(source, encoding="bytes")
        for key in (b"data", "data"):
            if key in batch:
                return len(batch[key])
        raise ValueError(f"{file_path.name}에 data 필드가 없습니다.")
