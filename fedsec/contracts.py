"""1번 담당 함수 사이에서 전달하는 Tensor 자료형. HTTP/JSON 필드는 포함하지 않는다."""

from dataclasses import dataclass, field
from typing import TypeAlias
import torch

TensorState: TypeAlias = dict[str, torch.Tensor]


@dataclass(frozen=True)
class ClientUpdate:
    client_id: int
    num_samples: int
    delta: TensorState


@dataclass(frozen=True)
class LocalTrainingResult:
    weights: TensorState
    delta: TensorState
    num_samples: int
    history: list[dict]


@dataclass(frozen=True)
class EvaluationResult:
    accuracy: float
    loss: float
    num_samples: int
    correct_predictions: int
    per_class: list[dict] = field(default_factory=list)
