"""1번 fedsec + 2번 security를 같은 브랜치에 넣었을 때 실행하는 통합 테스트.

보안 단독 브랜치에 1번 fedsec/이 아직 없다면 이 파일만 skip하고
기존 security 관련 테스트는 그대로 실행된다.
"""
from dataclasses import replace

import pytest
import torch

pytest.importorskip("fedsec")

from fedsec.config import DefenseConfig
from fedsec.contracts import ClientUpdate, RoundContext
from fedsec.tensors import state_hash
from security.integration import aggregate_with_defense


def _fixture(deltas, samples=None):
    """세 개 이상의 클라이언트가 공통 시작 모델을 쓰는 작은 예제."""
    global_state = {
        "layer.weight": torch.tensor([10.0, 20.0], dtype=torch.float32),
        "layer.bias": torch.tensor([1.0], dtype=torch.float32),
    }
    context = RoundContext("exp", 1, 0, tuple(range(len(deltas))))
    h = state_hash(global_state)
    samples = samples or [1] * len(deltas)
    updates = [
        ClientUpdate("exp", i, 1, 0, samples[i], h, {
            "layer.weight": torch.tensor([weight, 0.0], dtype=torch.float32),
            "layer.bias": torch.tensor([bias], dtype=torch.float32),
        })
        for i, (weight, bias) in enumerate(deltas)
    ]
    return global_state, updates, context


def _defense(method: str, bound: float) -> DefenseConfig:
    return DefenseConfig(enabled=True, type=method, parameters={"max_norm": bound})


def test_norm_clipping_uses_global_l2_across_all_layers_and_weighted_fedavg():
    state, updates, context = _fixture([(3.0, 4.0), (0.0, 0.0), (0.0, 0.0)])
    next_state = aggregate_with_defense(state, updates, context, _defense("norm_clipping", 2.5))
    # 첫 클라이언트의 전체 노름은 sqrt(3²+4²)=5; 클리핑 후 (1.5, 2.0).
    torch.testing.assert_close(next_state["layer.weight"], torch.tensor([10.5, 20.0]))
    torch.testing.assert_close(next_state["layer.bias"], torch.tensor([1.0 + 2.0 / 3]))
    torch.testing.assert_close(updates[0].delta["layer.weight"], torch.tensor([3.0, 0.0]))
    torch.testing.assert_close(state["layer.bias"], torch.tensor([1.0]))


def test_norm_clipping_respects_sample_count_for_fedavg():
    state, updates, context = _fixture([(1.0, 0.0), (3.0, 0.0), (-1.0, 0.0)], [1, 2, 1])
    next_state = aggregate_with_defense(state, updates, context, _defense("norm_clipping", 100.0))
    torch.testing.assert_close(next_state["layer.weight"], torch.tensor([11.5, 20.0]))


def test_clipping_median_even_number_uses_middle_two_and_not_fedavg():
    state, updates, context = _fixture([(-10.0, 0.0), (1.0, 0.0), (2.0, 0.0), (100.0, 0.0)])
    next_state = aggregate_with_defense(state, updates, context, _defense("coordinate_median", 1000.0))
    torch.testing.assert_close(next_state["layer.weight"], torch.tensor([11.5, 20.0]))
    torch.testing.assert_close(next_state["layer.bias"], torch.tensor([1.0]))


def test_less_than_three_accepted_updates_does_not_change_model():
    state, updates, context = _fixture([(100.0, 0.0), (200.0, 0.0)])
    next_state = aggregate_with_defense(state, updates, context, _defense("norm_clipping", 1.0))
    for key in state:
        torch.testing.assert_close(next_state[key], state[key], atol=0, rtol=0)
        assert next_state[key].data_ptr() != state[key].data_ptr()


def test_rejects_stale_round_and_invalid_clip_norm():
    state, updates, context = _fixture([(1.0, 0.0)] * 3)
    with pytest.raises(ValueError):
        aggregate_with_defense(state, [replace(updates[0], model_version=2), *updates[1:]], context,
                               _defense("norm_clipping", 1.0))
    for bad_norm in (0.0, -1.0, float("nan"), float("inf"), True):
        with pytest.raises(ValueError):
            aggregate_with_defense(state, updates, context,
                                   DefenseConfig(enabled=True, type="norm_clipping", parameters={"max_norm": bad_norm}))
    with pytest.raises(ValueError):
        aggregate_with_defense(state, updates, context,
                               DefenseConfig(enabled=True, type="coordinate_median", parameters={}))


def test_disabled_defense_falls_back_to_weighted_fedavg():
    state, updates, context = _fixture([(1.0, 0.0), (3.0, 0.0), (-1.0, 0.0)], [1, 2, 1])
    next_state = aggregate_with_defense(state, updates, context, DefenseConfig())
    torch.testing.assert_close(next_state["layer.weight"], torch.tensor([11.5, 20.0]))
