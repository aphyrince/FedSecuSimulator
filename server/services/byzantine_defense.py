"""내부자 공격 업데이트에 대한 방어 집계 서비스."""

from typing import Any


def apply_norm_clipping(
    updates: list[dict[str, Any]], max_norm: float
) -> list[dict[str, Any]]:
    """각 업데이트의 model_delta를 최대 노름 이하로 축소한다."""
    clipped = []
    max_norm = max(float(max_norm), 0.0)
    for update in updates:
        item = dict(update)
        delta = item.get("update", {}).get("model_delta", 0.0)
        delta = float(delta) if isinstance(delta, (int, float)) else 0.0
        norm = abs(delta)
        factor = min(1.0, max_norm / norm) if norm > 0 else 0.0
        item["update"] = {
            **item.get("update", {}),
            "model_delta": delta * factor,
            "defense": "norm_clipping",
            "clip_factor": factor,
        }
        local_model = item.get("local_model")
        if isinstance(local_model, dict) and isinstance(local_model.get("weights"), list):
            item["local_model"] = {
                **local_model,
                "weights": _scale_numeric_values(local_model["weights"], factor),
            }
        clipped.append(item)
    return clipped


def _scale_numeric_values(value: Any, factor: float) -> Any:
    if isinstance(value, list):
        return [_scale_numeric_values(item, factor) for item in value]
    if isinstance(value, dict):
        return {
            key: _scale_numeric_values(item, factor)
            for key, item in value.items()
        }
    if isinstance(value, (int, float)):
        return float(value) * factor
    return value


def coordinate_median(updates: list[dict[str, Any]]) -> float:
    """현재 기본 숫자 업데이트의 좌표별 중앙값을 계산한다."""
    values = sorted(
        float(update["update"]["model_delta"])
        for update in updates
        if isinstance(update.get("update", {}).get("model_delta"), (int, float))
    )
    if not values:
        return 0.0
    middle = len(values) // 2
    if len(values) % 2:
        return values[middle]
    return (values[middle - 1] + values[middle]) / 2.0
