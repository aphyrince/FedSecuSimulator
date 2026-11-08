"""Tensor 상태 검증/복사. 모든 계산 결과는 독립된 CPU float32 상태로 반환한다."""

import hashlib
import torch
from fedsec.contracts import TensorState


def copy_state(state: TensorState, reference: TensorState | None = None) -> TensorState:
    if not isinstance(state, dict) or not state:
        raise ValueError("Expected a nonempty tensor state")
    if reference is not None and state.keys() != reference.keys():
        raise ValueError("Parameter names differ")
    result = {}
    for name, value in state.items():
        if not isinstance(name, str) or not isinstance(value, torch.Tensor) or not value.is_floating_point():
            raise ValueError("State must contain named floating tensors")
        tensor = value.detach().to(device="cpu", dtype=torch.float32).clone()
        if not tensor.numel() or not torch.isfinite(tensor).all().item():
            raise ValueError("Empty or nonfinite tensor")
        if reference is not None and tensor.shape != reference[name].shape:
            raise ValueError(f"Parameter shape mismatch: {name}")
        result[name] = tensor
    return result


def extract_update(before: TensorState, after: TensorState) -> TensorState:
    before = copy_state(before)
    after = copy_state(after, before)
    return copy_state({key: after[key] - before[key] for key in before})


def state_hash(state: TensorState) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(copy_state(state).items()):
        digest.update(f"{name}:{tuple(tensor.shape)}:float32:".encode())
        digest.update(tensor.contiguous().numpy().tobytes())
    return digest.hexdigest()
