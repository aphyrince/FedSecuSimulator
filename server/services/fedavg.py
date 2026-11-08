"""기존 REST 응답 키를 유지하는 텐서 기반 FedAvg 및 좌표별 중앙값 집계."""

from copy import deepcopy
from typing import Any, TypedDict
import torch

from common.model import decode_weights, encode_weights
from common.updates import update_delta, validate_updates
from fedsec.contracts import ClientUpdate
from fedsec.aggregation import aggregate_fedavg as aggregate_core


class FedAvgResult(TypedDict):
    global_model: dict[str, Any]
    total_samples: int
    client_weights: dict[str, float]


def _accepted(global_model: dict, client_updates: list[dict]) -> list[dict]:
    accepted = [item for item in client_updates if item.get("valid") is True]
    return validate_updates(global_model, accepted)


def _result(global_model: dict, delta: dict, updates: list[dict], weights: list[float]) -> FedAvgResult:
    initial = decode_weights(global_model["weights"])
    updated = deepcopy(global_model)
    updated["weights"] = encode_weights({key: initial[key].double() + delta[key] for key in initial})
    updated["model_version"] = int(global_model["model_version"]) + 1
    return {"global_model": updated, "total_samples": sum(item["num_samples"] for item in updates),
            "client_weights": {str(item["client_id"]): weight for item, weight in zip(updates, weights)}}


def aggregate_fedavg(global_model: dict, client_updates: list[dict]) -> FedAvgResult:
    """각 업데이트에 표본 수 비율을 곱해 합산한 뒤 시작 전역 모델에 더한다."""
    updates = _accepted(global_model, client_updates)
    total = sum(item["num_samples"] for item in updates)
    weights = [item["num_samples"] / total for item in updates]
    tensors = [ClientUpdate(item["client_id"], item["num_samples"], update_delta(global_model, item))
               for item in updates]
    updated = deepcopy(global_model)
    updated["weights"] = encode_weights(aggregate_core(decode_weights(global_model["weights"]), tensors))
    updated["model_version"] = int(global_model["model_version"]) + 1
    return {"global_model": updated, "total_samples": total,
            "client_weights": {str(item["client_id"]): weight for item, weight in zip(updates, weights)}}


def aggregate_coordinate_median(global_model: dict, client_updates: list[dict]) -> FedAvgResult:
    """참여자 수가 짝수이면 가운데 두 좌표값의 평균을 사용한다."""
    updates = _accepted(global_model, client_updates)
    deltas = [update_delta(global_model, item) for item in updates]
    result = {}
    size = len(deltas)
    for key in deltas[0]:
        ordered = torch.stack([delta[key].double() for delta in deltas]).sort(dim=0).values
        result[key] = (ordered[(size - 1) // 2] + ordered[size // 2]) / 2
    return _result(global_model, result, updates, [1 / size] * size)
