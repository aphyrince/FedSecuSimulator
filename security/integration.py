"""2번 보안 집계 함수를 1번 fedsec 연합학습 API에 연결하는 어댑터.

1번: fedsec.simulation.simulate(..., aggregation_fn=aggregate_with_defense)
2번: security.defenses.clip_update / coordinate_median

보안 기능 자체는 security/defenses.py에서 구현한다. 이 파일에서는 오직
자료형(dict ↔ list[Tensor]) 변환, 유효성 검증, 집계, 모델 갱신을 수행한다.
"""
from __future__ import annotations

import math

import torch

from security.defenses import clip_update, coordinate_median


def _get_clip_bound(defense_config) -> float:
    """설정의 parameters['max_norm']을 필수 클리핑 상한으로 읽는다."""
    params = defense_config.parameters
    if not isinstance(params, dict) or "max_norm" not in params:
        raise ValueError("defense.parameters.max_norm을 설정해야 합니다.")
    max_norm = params["max_norm"]
    if isinstance(max_norm, bool) or not isinstance(max_norm, (int, float)):
        raise ValueError("max_norm은 0보다 큰 유한한 숫자여야 합니다.")
    max_norm = float(max_norm)
    if not math.isfinite(max_norm) or max_norm <= 0:
        raise ValueError("max_norm은 0보다 큰 유한한 숫자여야 합니다.")
    return max_norm


def aggregate_with_defense(global_state, updates, context, defense_config):
    """1번의 aggregation_fn 확장 지점에 그대로 연결한다.

    인자:
      global_state: 1번 fedsec의 dict[str, torch.Tensor]
      updates: 1번 fedsec.contracts.ClientUpdate 목록 (델타 = local-global)
      context: fedsec.contracts.RoundContext
      defense_config: fedsec.config.DefenseConfig
    반환: 집계 후 새 전역 모델 dict[str, torch.Tensor]

    방어 모드:
      norm_clipping     : 클리핑 -> 표본 수 가중 FedAvg
      coordinate_median : 클리핑 -> (비가중) 좌표별 중앙값

    fedsec의 모듈은 함수 안에서 불러와, security 브랜치에 fedsec이 아직
    없더라도 기존 20개 단위 테스트를 실행할 수 있다.
    """
    from fedsec.aggregation import fedavg, validate_updates
    from fedsec.tensors import clone_state, validate_state

    # 실제 연합학습은 잘못된 라운드/버전/해시/파라미터를 집계하면 안 된다.
    validate_updates(global_state, updates, context)

    # 과제 운영 규칙: 채택 업데이트가 3개 미만이면 모델 가중치는 갱신하지 않는다.
    # 모델 버전 유지/라운드 상태 기록은 호출자(3번 서버)가 별도로 처리해야 한다.
    if len(updates) < 3:
        return clone_state(global_state)

    if not defense_config.enabled or defense_config.type == "none":
        return fedavg(global_state, updates, context)

    if defense_config.type not in ("norm_clipping", "coordinate_median"):
        raise ValueError(f"지원하지 않는 방어 방식: {defense_config.type}")
    max_norm = _get_clip_bound(defense_config)

    # 딕셔너리 이름/순서가 다를 위험을 없애기 위해 global_state 이름을 기준으로 처리.
    names = list(global_state.keys())
    clipped_updates = []
    for update in updates:
        layers = [update.delta[name] for name in names]
        clipped_layers = clip_update(layers, max_norm=max_norm)
        clipped_delta = dict(zip(names, clipped_layers))
        # 클라이언트 ID/표본 수/라운드/모델 해시는 그대로 유지.
        clipped_updates.append(update.with_delta(clipped_delta))

    if defense_config.type == "norm_clipping":
        # 표본 수가 다르면 1번 FedAvg가 원래 규칙대로 가중평균함.
        return fedavg(global_state, clipped_updates, context)

    # Clipping + Coordinate-wise Median: FedAvg를 또 적용하면 안 된다.
    median_layers = coordinate_median(
        [[update.delta[name] for name in names] for update in clipped_updates]
    )
    next_state = {}
    for name, median_delta in zip(names, median_layers):
        origin = global_state[name]
        next_state[name] = (
            origin.detach().cpu().double() + median_delta.detach().cpu().double()
        ).to(origin.dtype)
    validate_state(next_state, global_state)
    return next_state
