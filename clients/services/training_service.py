"""클라이언트 로컬 학습과 업데이트 생성 서비스."""

from typing import Any
from copy import deepcopy

from clients.services.local_training import train_local_model
from common.model import weights_hash, weights_norm


def run_local_training(
    state,
    client_id: int,
    global_model: dict[str, Any],
) -> dict[str, Any]:
    """전역 모델 수신부터 로컬 학습, 업데이트 추출까지 수행한다."""
    round_id = int(global_model.get("round", 1))
    training_result = train_local_model(
        global_model=global_model,
        dataset_range=state.dataset_range,
        training_config={**state.experiment_config, **global_model.get("training_config", {})},
        round_id=round_id,
        client_id=client_id,
    )
    extracted_update = {"values": training_result["model_delta"],
                        "update_norm": weights_norm(training_result["model_delta"])}
    training_record = {
        "round": round_id,
        "client_id": client_id,
        "status": "completed",
        "extracted_update": extracted_update,
        **training_result,
    }
    state.save_training(training_record)
    return training_record


def build_round_update(
    state,
    client_id: int,
    request_payload: dict[str, Any],
) -> dict[str, Any]:
    """학습된 로컬 업데이트를 중앙 서버 제출 형식으로 만든다."""
    round_id = int(request_payload.get("round", 1))
    training = state.latest_local_training
    has_round_training = training.get("round") == round_id
    if not has_round_training or training.get("base_model_hash") != weights_hash(state.global_model["weights"]):
        raise ValueError("No completed training for the current round/model")
    if round_id in state.expired_rounds:
        raise ValueError("The submission deadline has expired")
    extracted_update = training.get("extracted_update", {})
    model_delta = extracted_update["update_norm"]

    malicious = bool(request_payload.get("malicious", False))
    attacked = malicious and bool(request_payload.get("attack_enabled", False))
    return {
        "round": round_id,
        "experiment_id": state.global_model.get("experiment_id"),
        "base_model_version": training["base_model_version"],
        "base_model_hash": training["base_model_hash"],
        "client_id": client_id,
        "malicious": malicious,
        "valid": True,
        "num_samples": training.get("num_samples", 0) if has_round_training else 0,
        "local_model": deepcopy(training["local_model"]),
        "dataset_fingerprint": training["dataset_fingerprint"],
        "update": {
            "values": deepcopy(extracted_update["values"]),
            "definition": "local_final_minus_global_start",
            "model_delta": model_delta,
            "scale": 1.0,
            "attack_candidate": attacked,
            "update_norm": (
                extracted_update.get("update_norm", 0.0)
                if has_round_training
                else 0.0
            ),
        },
    }
