"""FedAvg 집계 인터페이스와 기본 JSON 모델 구현."""

from copy import deepcopy
from typing import Any, TypedDict


class FedAvgResult(TypedDict):
    """FedAvg 결과와 집계 메타데이터."""

    global_model: dict[str, Any]
    total_samples: int
    client_weights: dict[str, float]


def _weighted_average(values: list[Any], weights: list[float]) -> Any:
    if not values:
        return None
    if isinstance(values[0], dict):
        keys = sorted(set().union(*(item.keys() for item in values)))
        return {
            key: _weighted_average(
                [item.get(key, 0) for item in values],
                weights,
            )
            for key in keys
        }
    if isinstance(values[0], list):
        length = max(len(item) for item in values)
        return [
            _weighted_average(
                [item[index] if index < len(item) else 0 for item in values],
                weights,
            )
            for index in range(length)
        ]
    return sum(float(value) * weight for value, weight in zip(values, weights))


def aggregate_fedavg(
    global_model: dict[str, Any],
    client_updates: list[dict[str, Any]],
) -> FedAvgResult:
    """표본 수를 가중치로 사용해 클라이언트 로컬 모델을 FedAvg한다.

    실제 팀원이 구현할 FedAvg/PyTorch 코드는 이 함수의 반환 형식을 유지하고,
    `local_model`과 `num_samples`를 읽어 전역 모델을 갱신하면 된다.
    """
    valid_updates = [
        update
        for update in client_updates
        if update.get("valid", False)
        and update.get("local_model", {}).get("weights") is not None
        and update.get("num_samples", 0) > 0
    ]
    if not valid_updates:
        raise ValueError("No valid local models are available for FedAvg")

    weights = [
        float(update["num_samples"])
        for update in valid_updates
    ]
    total_samples = sum(weights)
    normalized_weights = [weight / total_samples for weight in weights]
    client_models = [update["local_model"]["weights"] for update in valid_updates]
    aggregated_weights = _weighted_average(client_models, normalized_weights)

    updated_model = deepcopy(global_model)
    updated_model["weights"] = aggregated_weights
    updated_model["model_version"] = int(global_model.get("model_version", 0)) + 1
    client_weight_map = {
        str(update["client_id"]): round(
            float(update["num_samples"]) / total_samples,
            8,
        )
        for update in valid_updates
    }
    return {
        "global_model": updated_model,
        "total_samples": int(total_samples),
        "client_weights": client_weight_map,
    }


def aggregate_coordinate_median(
    global_model: dict[str, Any],
    client_updates: list[dict[str, Any]],
) -> FedAvgResult:
    """숫자 가중치 벨 모델의 좌표별 중앙값을 계산한다."""
    valid_updates = [
        update for update in client_updates
        if update.get("valid", False)
        and update.get("local_model", {}).get("weights") is not None
    ]
    if not valid_updates:
        raise ValueError("No valid local models are available for median aggregation")

    def median(values: list[Any]) -> Any:
        if isinstance(values[0], dict):
            keys = sorted(set().union(*(item.keys() for item in values)))
            return {
                key: median([item.get(key, 0) for item in values])
                for key in keys
            }
        if isinstance(values[0], list):
            length = max(len(item) for item in values)
            return [
                median([item[index] if index < len(item) else 0 for item in values])
                for index in range(length)
            ]
        ordered = sorted(float(value) for value in values)
        middle = len(ordered) // 2
        if len(ordered) % 2:
            return ordered[middle]
        return (ordered[middle - 1] + ordered[middle]) / 2.0

    updated_model = deepcopy(global_model)
    updated_model["weights"] = median(
        [update["local_model"]["weights"] for update in valid_updates]
    )
    updated_model["model_version"] = int(global_model.get("model_version", 0)) + 1
    total_samples = sum(int(update.get("num_samples", 0)) for update in valid_updates)
    return {
        "global_model": updated_model,
        "total_samples": total_samples,
        "client_weights": {
            str(update["client_id"]): 1.0 / len(valid_updates)
            for update in valid_updates
        },
    }
