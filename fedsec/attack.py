"""1번 담당: 협력 공격자들의 변조 전 delta 단순평균에 배율을 곱한다."""

import torch
from fedsec.contracts import TensorState
from fedsec.tensors import copy_state
from fedsec.validation import finite_number


def collude_updates(clean_deltas: list[TensorState], scale: float = -5.0) -> TensorState:
    finite_number(scale, "attack_scale")
    if len(clean_deltas) < 2:
        raise ValueError("Collusion requires at least two clean updates")
    base = copy_state(clean_deltas[0])
    deltas = [copy_state(item, base) for item in clean_deltas]
    return copy_state({name: torch.stack([delta[name].double() for delta in deltas]).mean(0) * scale
                       for name in base})
