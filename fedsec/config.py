"""API와 명령줄 실행에서 함께 사용하는 학습 설정을 정의하고 검증한다."""

from dataclasses import asdict, dataclass, field
import os
import random

import numpy as np
import torch

from fedsec.augmentation import AUGMENTATIONS
from fedsec.validation import finite_number, positive_int
from fedsec.model import model_config as validate_model_config


@dataclass(frozen=True)
class TrainingSettings:
    local_epochs: int = 2
    batch_size: int = 32
    learning_rate: float = 0.001
    optimizer: str = "sgd"
    momentum: float = 0.0
    weight_decay: float = 0.0
    device: str = "cpu"
    num_workers: int = 0
    torch_threads: int = 1
    max_train_samples: int | None = None
    max_test_samples: int | None = None
    augmentation: str = "none"

    def __post_init__(self):
        for key in ("local_epochs", "batch_size", "torch_threads"):
            positive_int(getattr(self, key), key)
        positive_int(self.num_workers, "num_workers", 0)
        finite_number(self.learning_rate, "learning_rate", positive=True)
        for key in ("momentum", "weight_decay"):
            if finite_number(getattr(self, key), key) < 0:
                raise ValueError(f"{key} must be nonnegative")
        for key in ("max_train_samples", "max_test_samples"):
            if getattr(self, key) is not None:
                positive_int(getattr(self, key), key)
        if self.optimizer not in ("sgd", "adam") or self.device not in ("cpu", "cuda", "auto"):
            raise ValueError("Unsupported optimizer or device")
        if self.optimizer == "adam" and self.momentum != 0:
            raise ValueError("momentum is only supported by SGD")
        if self.augmentation not in AUGMENTATIONS:
            raise ValueError("Unsupported data augmentation")

    @classmethod
    def from_dict(cls, values: dict):
        return cls(**{key: values[key] for key in cls.__dataclass_fields__ if key in values})

    def to_dict(self) -> dict:
        values = asdict(self)
        if self.augmentation == "none":
            del values["augmentation"]
        return values


@dataclass(frozen=True)
class ExperimentSettings:
    """독립 실행의 공통 조건. 참여율/지연/방어 실험은 2·3번 담당 시스템에서 설정한다."""

    num_rounds: int = 10
    seed: int = 42
    training: TrainingSettings = field(default_factory=TrainingSettings)
    model_config: dict = field(default_factory=dict)
    malicious_clients: tuple[int, ...] = (0, 1)
    attack_scale: float = -5.0

    def __post_init__(self):
        positive_int(self.num_rounds, "num_rounds")
        positive_int(self.seed, "seed", 0)
        if not isinstance(self.training, TrainingSettings):
            raise ValueError("training must be TrainingSettings")
        object.__setattr__(self, "model_config", validate_model_config(self.model_config))
        ids = tuple(positive_int(item, "malicious client ID", 0) for item in self.malicious_clients)
        if len(set(ids)) != len(ids):
            raise ValueError("Duplicate malicious client ID")
        object.__setattr__(self, "malicious_clients", ids)
        finite_number(self.attack_scale, "attack_scale")

    @classmethod
    def from_dict(cls, values: dict):
        unknown = values.keys() - cls.__dataclass_fields__.keys()
        if unknown:
            raise ValueError(f"Unknown experiment settings: {sorted(unknown)}")
        training = values.get("training", {})
        unknown_training = training.keys() - TrainingSettings.__dataclass_fields__.keys()
        if unknown_training:
            raise ValueError(f"Unknown training settings: {sorted(unknown_training)}")
        return cls(**{**values, "training": TrainingSettings.from_dict(training)})

    def to_dict(self) -> dict:
        return asdict(self)


def setup_runtime(settings: TrainingSettings, seed: int) -> torch.device:
    positive_int(seed, "seed", 0)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_num_threads(settings.torch_threads)
    name = settings.device
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable; select device=cpu")
    return torch.device(name)
