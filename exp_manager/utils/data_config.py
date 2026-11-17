from pathlib import Path
from os import environ

from dotenv import load_dotenv

from exp_manager.utils.project_root_path import PROJECT_ROOT_PATH


ENV_PATH = Path(__file__).resolve().parent.parent / ".env"


def read_data_config() -> dict:
    load_dotenv(ENV_PATH, override=False)
    source_dir = (
        environ.get("SOURCE_DIR")
        or environ.get("DATA_SOURCE_DIR")
        or "source"
    )
    train_files = (
        environ.get("TRAIN_DATA_FILES")
        or environ.get("TRAIN_DATA_FILE")
        or environ.get("TRAIN_FILENAME")
        or environ.get("CIFAR_TRAIN_FILE")
        or ""
    )
    test_files = (
        environ.get("TEST_DATA_FILES")
        or environ.get("TEST_DATA_FILE")
        or environ.get("TEST_FILENAME")
        or environ.get("CIFAR_TEST_FILE")
        or ""
    )
    return {
        "source_dir": source_dir,
        "train_files": [name.strip() for name in train_files.split(",") if name.strip()],
        "test_files": [name.strip() for name in test_files.split(",") if name.strip()],
        "test_ratio": float(environ.get("CIFAR_TEST_RATIO", "0.2")),
        "source_path": str((PROJECT_ROOT_PATH / source_dir).resolve()),
    }
