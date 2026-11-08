"""변조 전 delta를 평균하고 공격 배율을 곱한 뒤 제출 모델을 다시 구성한다."""

from fedsec.attack import collude_updates as collude_core
from common.model import finite_number, positive_int
from common.updates import update_delta, validate_updates, with_delta


def collude_updates(global_model: dict, own_update: dict, peer_updates: list[dict],
                    scale: float = -5.0, min_colluders: int = 2) -> dict:
    finite_number(scale, "attack_scale")
    positive_int(min_colluders, "min_colluders")
    updates = [own_update, *peer_updates]
    validate_updates(global_model, updates)
    if len(updates) < min_colluders:
        result = with_delta(global_model, own_update, update_delta(global_model, own_update))
        result["update"].update(attack_applied=False, attack_reason="insufficient_colluders")
        return result
    deltas = [update_delta(global_model, item) for item in updates]
    result = with_delta(global_model, own_update, collude_core(deltas, scale))
    result["malicious"] = True
    result["update"].update(scale=scale, attack_applied=True, collusion_peer_count=len(peer_updates))
    return result
