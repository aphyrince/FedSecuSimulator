"""Strict JSON configuration. Loading a configuration never starts training."""

from dataclasses import asdict, dataclass, field, fields
import json
import math
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def integer(name: str, value: Any, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name}: {minimum} 이상의 정수가 필요합니다.")
    return value


def number(name: str, value: Any, minimum=None, *, positive=False) -> float:
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"{name}: 유한한 숫자가 필요합니다.")
    if (positive and value <= 0) or (minimum is not None and value < minimum):
        raise ValueError(f"{name}: 허용 범위를 벗어났습니다.")
    return float(value)


@dataclass(frozen=True)
class ModelConfig:
    name: str = "small_cnn"
    channels: tuple[int, int] = (32, 64)
    hidden_size: int = 128

    def __post_init__(self):
        if self.name != "small_cnn" or len(self.channels) != 2:
            raise ValueError("지원 모델은 합성곱 2층의 small_cnn입니다.")
        for channel in self.channels:
            integer("model.channels", channel, 1)
        integer("model.hidden_size", self.hidden_size, 1)


@dataclass(frozen=True)
class TrainingConfig:
    local_epochs: int = 2
    batch_size: int = 32
    learning_rate: float = 0.001
    optimizer: str = "sgd"
    momentum: float = 0.0
    weight_decay: float = 0.0

    def __post_init__(self):
        integer("local_epochs", self.local_epochs, 1)
        integer("batch_size", self.batch_size, 1)
        number("learning_rate", self.learning_rate, positive=True)
        number("momentum", self.momentum, 0)
        number("weight_decay", self.weight_decay, 0)
        if self.optimizer not in ("sgd", "adam"):
            raise ValueError("optimizer는 sgd 또는 adam이어야 합니다.")
        if self.optimizer == "adam" and self.momentum != 0:
            raise ValueError("adam에는 momentum을 지정하지 않습니다.")


@dataclass(frozen=True)
class DataConfig:
    root: str = "data/raw"
    partition_file: str | None = None
    validation_size: int = 0
    normalization_mean: tuple[float, float, float] = (0.5, 0.5, 0.5)
    normalization_std: tuple[float, float, float] = (0.5, 0.5, 0.5)
    max_train_samples_per_client: int | None = None

    def __post_init__(self):
        if not isinstance(self.root, str) or not self.root:
            raise ValueError("data.root가 필요합니다.")
        if self.partition_file is not None and (not isinstance(self.partition_file, str) or not self.partition_file.strip()):
            raise ValueError("partition_file은 경로 또는 null이어야 합니다.")
        integer("validation_size", self.validation_size)
        if self.validation_size >= 50000:
            raise ValueError("validation_size는 50000 미만이어야 합니다.")
        if len(self.normalization_mean) != 3 or len(self.normalization_std) != 3:
            raise ValueError("CIFAR-10 정규화는 RGB 채널별 값 3개가 필요합니다.")
        for value in self.normalization_mean:
            number("normalization_mean", value)
        for value in self.normalization_std:
            number("normalization_std", value, positive=True)
        if self.max_train_samples_per_client is not None:
            integer("max_train_samples_per_client", self.max_train_samples_per_client, 1)


@dataclass(frozen=True)
class EvaluationConfig:
    split: str = "test"
    batch_size: int = 256
    max_samples: int | None = None

    def __post_init__(self):
        if self.split not in ("validation", "test"):
            raise ValueError("evaluation.split은 validation 또는 test여야 합니다.")
        integer("evaluation.batch_size", self.batch_size, 1)
        if self.max_samples is not None:
            integer("evaluation.max_samples", self.max_samples, 1)


@dataclass(frozen=True)
class AttackConfig:
    enabled: bool = True
    type: str = "insider_collusion"
    malicious_clients: tuple[int, ...] = (2, 7)
    scale: float = -5.0
    min_colluders: int = 2
    missing_peer_policy: str = "skip_attack"

    def __post_init__(self):
        if type(self.enabled) is not bool or self.type != "insider_collusion":
            raise ValueError("공격은 enabled(bool), type=insider_collusion을 사용합니다.")
        for client_id in self.malicious_clients:
            integer("malicious_clients", client_id)
        if len(set(self.malicious_clients)) != len(self.malicious_clients):
            raise ValueError("malicious_clients에 중복 ID가 있습니다.")
        number("attack.scale", self.scale)
        integer("min_colluders", self.min_colluders, 1)
        if self.enabled and len(self.malicious_clients) < self.min_colluders:
            raise ValueError("악성 클라이언트 수가 min_colluders보다 적습니다.")
        if self.missing_peer_policy not in ("skip_attack", "available_only", "error"):
            raise ValueError("missing_peer_policy는 skip_attack/available_only/error입니다.")


@dataclass(frozen=True)
class DefenseConfig:
    enabled: bool = False
    type: str = "none"
    parameters: dict = field(default_factory=dict)

    def __post_init__(self):
        if type(self.enabled) is not bool or not isinstance(self.parameters, dict):
            raise ValueError("방어 설정의 enabled와 parameters 형식을 확인하세요.")
        if self.type not in ("none", "norm_clipping", "coordinate_median"):
            raise ValueError("지원 방어 연결 이름: none/norm_clipping/coordinate_median")
        if self.enabled and self.type == "none":
            raise ValueError("방어 활성화 시 type을 지정하세요.")


@dataclass(frozen=True)
class RuntimeConfig:
    device: str = "cpu"
    num_workers: int = 0
    torch_threads: int = 2
    output_root: str = "outputs"
    deterministic: bool = True

    def __post_init__(self):
        if self.device not in ("cpu", "cuda", "auto"):
            raise ValueError("device는 cpu/cuda/auto입니다.")
        integer("num_workers", self.num_workers)
        integer("torch_threads", self.torch_threads, 1)
        if type(self.deterministic) is not bool:
            raise ValueError("deterministic은 bool이어야 합니다.")
        if not isinstance(self.output_root, str) or not self.output_root:
            raise ValueError("runtime.output_root가 필요합니다.")


@dataclass(frozen=True)
class ExperimentConfig:
    experiment_id: str = "cifar10_001"
    seed: int = 42
    num_clients: int = 10
    num_rounds: int = 20
    participation_rate: float = 1.0
    participation_schedule: dict[str, list[int]] = field(default_factory=dict)
    initial_checkpoint: str | None = None
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    attack: AttackConfig = field(default_factory=AttackConfig)
    defense: DefenseConfig = field(default_factory=DefenseConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)

    def __post_init__(self):
        if not isinstance(self.experiment_id, str) or not self.experiment_id.strip():
            raise ValueError("experiment_id가 필요합니다.")
        integer("seed", self.seed)
        if self.seed >= 2**32:
            raise ValueError("seed는 2**32 미만이어야 합니다.")
        integer("num_clients", self.num_clients, 1)
        integer("num_rounds", self.num_rounds, 1)
        if not 0 < number("participation_rate", self.participation_rate) <= 1:
            raise ValueError("participation_rate는 0 초과 1 이하여야 합니다.")
        if self.initial_checkpoint is not None and (not isinstance(self.initial_checkpoint, str) or not self.initial_checkpoint.strip()):
            raise ValueError("initial_checkpoint는 경로 또는 null이어야 합니다.")
        if self.attack.enabled and any(i >= self.num_clients for i in self.attack.malicious_clients):
            raise ValueError("악성 ID가 num_clients 범위를 벗어났습니다.")
        if self.evaluation.split == "validation" and not self.data.validation_size and not self.data.partition_file:
            raise ValueError("검증 평가에는 validation_size 또는 외부 partition_file이 필요합니다.")
        if not isinstance(self.participation_schedule, dict):
            raise ValueError("participation_schedule은 라운드 번호를 키로 하는 객체입니다.")
        for key, ids in self.participation_schedule.items():
            if not isinstance(key, str) or not key.isdigit() or not 1 <= int(key) <= self.num_rounds or str(int(key)) != key:
                raise ValueError("참여 일정의 라운드 번호가 올바르지 않습니다.")
            if not isinstance(ids, list) or not ids:
                raise ValueError("라운드 참여 목록은 비어 있지 않은 ID 목록이어야 합니다.")
            for client_id in ids:
                integer("participation_schedule client_id", client_id)
                if client_id >= self.num_clients:
                    raise ValueError("참여 ID가 num_clients 범위를 벗어났습니다.")
            if len(set(ids)) != len(ids):
                raise ValueError("라운드 참여 목록에 중복 ID가 있습니다.")

    def to_dict(self) -> dict:
        return asdict(self)


def project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _merge(base: dict, override: dict) -> dict:
    result = dict(base)
    for key, value in override.items():
        result[key] = _merge(result[key], value) if isinstance(value, dict) and isinstance(result.get(key), dict) else value
    return result


def _unique_keys(pairs: list[tuple[str, Any]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"설정 파일에 중복된 키가 있습니다: {key}")
        result[key] = value
    return result


def _read(path: Path, stack: tuple[Path, ...] = ()) -> dict:
    path = path.resolve()
    if path in stack:
        raise ValueError("설정 extends에 순환 참조가 있습니다.")
    payload = json.loads(path.read_text(encoding="utf-8-sig"), object_pairs_hook=_unique_keys)
    # Reject NaN/Infinity even inside role-2 parameter dictionaries.
    json.dumps(payload, allow_nan=False)
    if not isinstance(payload, dict):
        raise ValueError("설정 파일의 최상위 값은 객체여야 합니다.")
    parent = payload.pop("extends", None)
    if parent is not None:
        if not isinstance(parent, str):
            raise ValueError("extends에는 설정 파일 경로를 지정하세요.")
        payload = _merge(_read(path.parent / parent, (*stack, path)), payload)
    return payload


def _construct(cls, values: dict):
    if not isinstance(values, dict):
        raise ValueError(f"{cls.__name__}: 객체가 필요합니다.")
    unknown = set(values) - {f.name for f in fields(cls)}
    if unknown:
        raise ValueError(f"{cls.__name__}: 알 수 없는 설정 {sorted(unknown)}")
    return cls(**values)


def load_config(path: str | Path) -> ExperimentConfig:
    values = _read(Path(path))
    for key, cls in (("model", ModelConfig), ("training", TrainingConfig), ("data", DataConfig),
                     ("evaluation", EvaluationConfig), ("attack", AttackConfig),
                     ("defense", DefenseConfig), ("runtime", RuntimeConfig)):
        if key in values:
            section = values[key]
            if not isinstance(section, dict):
                raise ValueError(f"{key}: 객체가 필요합니다.")
            for name in ("channels", "normalization_mean", "normalization_std", "malicious_clients"):
                if name in section:
                    if not isinstance(section[name], list):
                        raise ValueError(f"{key}.{name}: 배열이 필요합니다.")
                    section[name] = tuple(section[name])
            values[key] = _construct(cls, section)
    return _construct(ExperimentConfig, values)
