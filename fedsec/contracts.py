"""Algorithm-side contracts. HTTP handlers and file downloads belong to role 3."""

from dataclasses import dataclass, replace
from collections.abc import Mapping
from .config import integer, number
from .tensors import TensorState, clone_state, validate_state

UPDATE_DEFINITION = "local_final_minus_global_start"


@dataclass(frozen=True)
class RoundContext:
    experiment_id: str
    round_id: int
    model_version: int
    selected_clients: tuple[int, ...]

    def __post_init__(self):
        if not isinstance(self.experiment_id, str) or not self.experiment_id.strip():
            raise ValueError("experiment_id가 필요합니다.")
        integer("round_id", self.round_id, 1)
        integer("model_version", self.model_version)
        if not self.selected_clients or len(set(self.selected_clients)) != len(self.selected_clients):
            raise ValueError("선정 클라이언트는 중복 없는 목록이어야 합니다.")
        for client_id in self.selected_clients:
            integer("selected_clients", client_id)


@dataclass(frozen=True)
class TrainRequest:
    experiment_id: str
    round_id: int
    model_version: int
    local_epochs: int
    batch_size: int
    learning_rate: float
    model_url: str = ""

    def __post_init__(self):
        if not isinstance(self.experiment_id, str) or not self.experiment_id.strip():
            raise ValueError("experiment_id가 필요합니다.")
        integer("round_id", self.round_id, 1)
        integer("model_version", self.model_version)
        integer("local_epochs", self.local_epochs, 1)
        integer("batch_size", self.batch_size, 1)
        number("learning_rate", self.learning_rate, positive=True)
        if not isinstance(self.model_url, str):
            raise ValueError("model_url은 문자열이어야 합니다.")

    @classmethod
    def from_api(cls, payload: Mapping, *, experiment_id: str | None = None):
        if not isinstance(payload, Mapping):
            raise ValueError("/train JSON 객체가 필요합니다.")
        required = ("round_id", "model_version", "local_epochs", "batch_size", "learning_rate")
        missing = set(required) - set(payload)
        if missing:
            raise ValueError(f"/train 필수 필드 누락: {sorted(missing)}")
        received_id = payload.get("experiment_id", experiment_id)
        if experiment_id is not None and received_id != experiment_id:
            raise ValueError("/train experiment_id가 현재 실험과 다릅니다.")
        return cls(experiment_id=received_id, **{key: payload[key] for key in required},
                   model_url=payload.get("model_url", ""))


@dataclass(frozen=True)
class ClientUpdate:
    experiment_id: str
    client_id: int
    round_id: int
    model_version: int
    num_samples: int
    base_model_hash: str
    delta: TensorState

    def __post_init__(self):
        if not isinstance(self.experiment_id, str) or not self.experiment_id.strip():
            raise ValueError("업데이트 experiment_id가 필요합니다.")
        integer("client_id", self.client_id)
        integer("round_id", self.round_id, 1)
        integer("model_version", self.model_version)
        integer("num_samples", self.num_samples, 1)
        if not isinstance(self.base_model_hash, str) or len(self.base_model_hash) != 64 or any(c not in "0123456789abcdef" for c in self.base_model_hash):
            raise ValueError("base_model_hash는 SHA-256 문자열이어야 합니다.")
        validate_state(self.delta)

    def metadata(self) -> dict:
        return {"experiment_id": self.experiment_id, "client_id": self.client_id,
                "round_id": self.round_id, "model_version": self.model_version,
                "num_samples": self.num_samples, "base_model_hash": self.base_model_hash,
                "definition": UPDATE_DEFINITION}

    def with_delta(self, delta: TensorState):
        return replace(self, delta=clone_state(delta))

    def validate_for(self, context: RoundContext, reference: TensorState, base_hash: str):
        if (self.experiment_id, self.round_id, self.model_version) != (
                context.experiment_id, context.round_id, context.model_version):
            raise ValueError("다른 실험/라운드/모델 버전의 업데이트입니다.")
        if self.client_id not in context.selected_clients:
            raise ValueError("선정되지 않은 클라이언트의 업데이트입니다.")
        if self.base_model_hash != base_hash:
            raise ValueError("서로 다른 시작 모델로 계산한 업데이트입니다.")
        validate_state(self.delta, reference)
