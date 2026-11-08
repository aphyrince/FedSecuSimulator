"""1번 담당: 전역 모델 + 표본 수 가중 평균 delta. 악성 ID와 HTTP 상태는 알지 못한다."""

import torch
from fedsec.contracts import ClientUpdate, TensorState
from fedsec.tensors import copy_state
from fedsec.validation import positive_int


def aggregate_fedavg(global_weights: TensorState, updates: list[ClientUpdate]) -> TensorState:
    base = copy_state(global_weights)
    if not updates:
        raise ValueError("No updates to aggregate")
    ids = [positive_int(item.client_id, "client_id", 0) for item in updates]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate client update")
    total = sum(positive_int(item.num_samples, "num_samples") for item in updates)
    deltas = [copy_state(item.delta, base) for item in updates]
    result = {}
    for name in base:
        average = sum((delta[name].double() * (item.num_samples / total)
                       for delta, item in zip(deltas, updates)), torch.zeros_like(base[name], dtype=torch.float64))
        result[name] = base[name].double() + average
    return copy_state(result, base)
