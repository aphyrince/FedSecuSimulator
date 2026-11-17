"""REST 요청·응답 키를 유지하면서 실제 모델 delta를 검증하고 변환한다."""

from copy import deepcopy
import torch

from common.model import decode_weights, encode_weights, positive_int, weights_hash, weights_norm
from fedsec.model import SUPPORTED_ARCHITECTURES


def update_delta(global_model: dict, update: dict) -> dict:
    positive_int(update.get("client_id"), "client_id", 0)
    positive_int(update.get("num_samples"), "num_samples")
    if update.get("valid") is not True:
        raise ValueError("Invalid update")
    base = decode_weights(global_model["weights"])
    local = update.get("local_model", {})
    tensor_model = global_model.get("architecture") in SUPPORTED_ARCHITECTURES
    if tensor_model:
        positive_int(update.get("base_model_version"), "base_model_version", 0)
        if update.get("base_model_version") != global_model["model_version"]:
            raise ValueError("Update belongs to a different model version")
        if update.get("base_model_hash") != weights_hash(base):
            raise ValueError("Update belongs to a different starting model")
        if global_model.get("experiment_id") != update.get("experiment_id"):
            raise ValueError("Update belongs to a different experiment")
        for key in ("architecture", "model_config", "preprocessing"):
            if local.get(key) != global_model.get(key):
                raise ValueError(f"Local model {key} mismatch")
    final = decode_weights(local.get("weights"), base)
    delta = {key: final[key] - base[key] for key in base}
    submitted = update.get("update", {}).get("values")
    if tensor_model and submitted is None:
        raise ValueError("Missing tensor update.values")
    if submitted is not None:
        submitted = decode_weights(submitted, base)
        for name in base:
            if not torch.allclose(base[name] + submitted[name], final[name], rtol=1e-5, atol=1e-6):
                raise ValueError("Submitted delta does not match the local model")
        delta = submitted
    return delta


def validate_updates(global_model: dict, updates: list[dict]) -> list[dict]:
    if not updates:
        raise ValueError("No valid updates to aggregate")
    ids = [item.get("client_id") for item in updates]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate client update")
    rounds = set()
    for item in updates:
        positive_int(item.get("round"), "round")
        rounds.add(item["round"])
        update_delta(global_model, item)
    if len(rounds) != 1:
        raise ValueError("Mixed update rounds")
    return updates


def with_delta(global_model: dict, update: dict, delta: dict) -> dict:
    base = decode_weights(global_model["weights"])
    delta = decode_weights(delta, base)
    result = deepcopy(update)
    result["local_model"]["weights"] = encode_weights({key: base[key] + delta[key] for key in base})
    result["update"] = {**result.get("update", {}), "values": encode_weights(delta),
                        "model_delta": weights_norm(delta), "update_norm": weights_norm(delta),
                        "definition": "local_final_minus_global_start"}
    return result
