"""1번 담당: 전역 모델 + 표본 수 가중 평균 delta. 악성 ID와 HTTP 상태는 알지 못한다."""

import torch
from fedsec.contracts import ClientUpdate, TensorState
from fedsec.tensors import copy_state
from fedsec.validation import positive_int
from security.defenses import clip_update, coordinate_median

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

def clip_client_updates(
    global_weights: TensorState,
    updates: list[ClientUpdate],
    max_norm: float,
) -> list[ClientUpdate]:

    base = copy_state(global_weights)
    names = list(base.keys())
    clipped = []

    for item in updates:
        delta = copy_state(item.delta, base)

        layers = [delta[name] for name in names]

        clipped_layers = clip_update(
            layers,
            max_norm,
        )

        clipped_delta = copy_state(
            dict(zip(names, clipped_layers)),
            base,
        )

        clipped.append(
            ClientUpdate(
                item.client_id,
                item.num_samples,
                clipped_delta,
            )
        )

    return clipped

def aggregate_coordinate_median(
    global_weights: TensorState,
    updates: list[ClientUpdate],
) -> TensorState:

    base = copy_state(global_weights)

    if not updates:
        raise ValueError("No updates to aggregate")

    ids = [
        positive_int(item.client_id, "client_id", 0)
        for item in updates
    ]

    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate client update")

    for item in updates:
        positive_int(item.num_samples, "num_samples")

    names = list(base.keys())
    deltas = [
        copy_state(item.delta, base)
        for item in updates
    ]

    layers = [
        [delta[name].double() for name in names]
        for delta in deltas
    ]

    median_layers = coordinate_median(layers)

    result = {
        name: base[name].double() + median_layers[i]
        for i, name in enumerate(names)
    }

    return copy_state(result, base)