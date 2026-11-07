"""FedAvg receives submitted updates only; it has no malicious-ID input."""

import torch
from .contracts import ClientUpdate, RoundContext
from .tensors import TensorState, state_hash, validate_state


def validate_updates(global_state: TensorState, updates: list[ClientUpdate], context: RoundContext,
                     *, sample_counts: dict[int, int] | None = None) -> None:
    validate_state(global_state)
    if not updates:
        raise ValueError("업데이트가 없어 집계할 수 없습니다.")
    ids = [update.client_id for update in updates]
    if len(set(ids)) != len(ids):
        raise ValueError("같은 클라이언트의 업데이트가 중복 제출되었습니다.")
    base_hash = state_hash(global_state)
    for update in updates:
        update.validate_for(context, global_state, base_hash)
        if sample_counts is not None and sample_counts.get(update.client_id) != update.num_samples:
            raise ValueError("업데이트 표본 수가 매니저의 배정 기록과 다릅니다.")


def fedavg(global_state: TensorState, updates: list[ClientUpdate], context: RoundContext,
           *, sample_counts: dict[int, int] | None = None) -> TensorState:
    """W_next = W_start + sum(n_k / sum(n_j) * delta_k).

    Missing selected clients are allowed; normalize over received samples only.
    The server must additionally verify num_samples against its dataset registry.
    """
    validate_updates(global_state, updates, context, sample_counts=sample_counts)
    total_samples = sum(update.num_samples for update in updates)
    result = {}
    for name, original in global_state.items():
        average = torch.zeros_like(original, dtype=torch.float64, device="cpu")
        for update in updates:
            average.add_(update.delta[name].detach().cpu().double(), alpha=update.num_samples / total_samples)
        result[name] = (original.detach().cpu().double() + average).to(original.dtype)
    validate_state(result, global_state)
    return result
