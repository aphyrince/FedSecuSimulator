"""변조 전 delta를 평균하고 공격 배율을 곱한 뒤 제출 모델을 다시 구성한다."""

from fedsec.attack import collude_updates as collude_core
from common.model import finite_number, positive_int, weights_norm
from common.updates import update_delta, validate_updates, with_delta


def collude_updates(global_model: dict, own_update: dict, peer_updates: list[dict],
                    scale: float = -5.0, min_colluders: int = 2) -> dict:
    finite_number(scale, "attack_scale")
    positive_int(min_colluders, "min_colluders")
    updates = [own_update, *peer_updates]
    validate_updates(global_model, updates)
    deltas = [update_delta(global_model, item) for item in updates]
    averaged_delta = (
        update_delta(global_model, own_update)
        if len(deltas) == 1
        else collude_core(deltas, 1.0)
    )
    manipulated_delta = (
        {name: delta * scale for name, delta in deltas[0].items()}
        if len(deltas) == 1
        else collude_core(deltas, scale)
    )
    result = with_delta(global_model, own_update, manipulated_delta)
    result["malicious"] = True
    result["update"].update(
        scale=scale,
        attack_applied=True,
        attack_reason=(
            "single_malicious_client_self_scaled"
            if len(updates) == 1
            else None
        ),
        collusion_peer_count=len(peer_updates),
        norm_before_attack=weights_norm(averaged_delta),
        norm_after_attack=weights_norm(manipulated_delta),
        own_update_norm=weights_norm(update_delta(global_model, own_update)),
    )
    return result
