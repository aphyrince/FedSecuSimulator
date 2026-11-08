"""클라이언트 로컬 학습과 업데이트 생성 서비스."""

from typing import Any

from services.local_training import extract_model_update, train_local_model


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
        training_config=state.experiment_config,
        round_id=round_id,
        client_id=client_id,
    )
    extracted_update = extract_model_update(
        global_model,
        training_result["local_model"],
    )
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
    extracted_update = training.get("extracted_update", {})
    extracted_values = extracted_update.get("values", 0.82)
    model_delta = (
        extracted_values
        if has_round_training and isinstance(extracted_values, (int, float))
        else extracted_update.get("update_norm", 0.82)
        if has_round_training
        else 0.82
    )

    malicious = bool(request_payload.get("malicious", False))
    attacked = malicious and bool(request_payload.get("attack_enabled", False))
    return {
        "round": round_id,
        "client_id": client_id,
        "malicious": malicious,
        "valid": True,
        "num_samples": training.get("num_samples", 0) if has_round_training else 0,
        "local_model": training.get("local_model", {}) if has_round_training else {},
        "update": {
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
