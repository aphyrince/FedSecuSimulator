"""Shared update representation: finite, dense, floating-point CPU tensors."""

import hashlib
from collections.abc import Mapping
import torch

TensorState = dict[str, torch.Tensor]


def validate_state(state: Mapping, reference: Mapping | None = None) -> None:
    if not isinstance(state, Mapping) or not state:
        raise ValueError("모델/업데이트 텐서가 비어 있습니다.")
    if reference is not None and set(state) != set(reference):
        raise ValueError("모델과 업데이트의 가중치 이름이 다릅니다.")
    for name, tensor in state.items():
        if not isinstance(name, str) or not isinstance(tensor, torch.Tensor):
            raise ValueError("가중치는 문자열 이름과 Tensor로 구성해야 합니다.")
        if tensor.layout != torch.strided or not tensor.is_floating_point() or tensor.numel() == 0:
            raise ValueError(f"{name}: 비어 있지 않은 실수 밀집 텐서가 필요합니다.")
        if not torch.isfinite(tensor).all().item():
            raise ValueError(f"{name}: NaN/Inf가 포함되어 있습니다.")
        if reference is not None:
            other = reference[name]
            if tensor.shape != other.shape or tensor.dtype != other.dtype:
                raise ValueError(f"{name}: 텐서 모양/자료형이 모델과 다릅니다.")


def clone_state(state: Mapping) -> TensorState:
    validate_state(state)
    return {name: tensor.detach().cpu().clone() for name, tensor in state.items()}


def state_hash(state: Mapping) -> str:
    validate_state(state)
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        digest.update(f"{name}:{tuple(tensor.shape)}:{tensor.dtype}:".encode())
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def state_norm(state: Mapping) -> float:
    validate_state(state)
    return sum(t.detach().double().square().sum().item() for t in state.values()) ** 0.5
