from flask import Blueprint, current_app, jsonify, request

from services.local_training import train_local_model


experiment_bp = Blueprint("client_experiment", __name__)


@experiment_bp.post("/api/experiment/config")
def set_config():
    # 실험에서 사용할 라운드·참여·공격·방어 설정을 저장한다.
    app = current_app._get_current_object()
    app.extensions["experiment_config"] = request.get_json(silent=True) or {}
    return jsonify(
        status="configured",
        client_id=app.config["CLIENT_ID"],
        config=app.extensions["experiment_config"],
    )


@experiment_bp.get("/api/experiment/config")
def get_config():
    return jsonify(current_app.extensions["experiment_config"])


@experiment_bp.post("/api/model/global")
def receive_global_model():
    """중앙 서버가 전달한 전역 모델을 수신해 로컬 최신 모델로 저장한다."""
    payload = request.get_json(silent=True) or {}
    current_app.extensions["global_model"] = payload
    current_app.extensions["global_models"].append(payload)
    round_id = int(payload.get("round", 1))
    training_result = train_local_model(
        global_model=payload,
        dataset_range=current_app.extensions.get("dataset_range", {}),
        training_config=current_app.extensions.get("experiment_config", {}),
        round_id=round_id,
        client_id=current_app.config["CLIENT_ID"],
    )
    training_record = {
        "round": round_id,
        "client_id": current_app.config["CLIENT_ID"],
        "status": "completed",
        **training_result,
    }
    current_app.extensions["latest_local_training"] = training_record
    current_app.extensions["local_training_results"].append(training_record)
    return jsonify(
        status="received",
        client_id=current_app.config["CLIENT_ID"],
        model_version=payload.get("model_version"),
        training=training_record,
    )


@experiment_bp.get("/api/model/global")
def get_global_model():
    return jsonify(current_app.extensions.get("global_model", {}))


@experiment_bp.post("/api/round/execute")
def execute_round():
    """실험 관리자가 요청한 로컬 라운드 업데이트를 생성한다."""
    payload = request.get_json(silent=True) or {}
    malicious = payload.get("malicious", False)
    attacked = malicious and payload.get("attack_enabled", False)
    round_id = payload.get("round", 1)
    latest_training = current_app.extensions.get("latest_local_training", {})
    model_delta = (
        latest_training.get("model_delta", 0.82)
        if latest_training.get("round") == round_id
        else 0.82
    )
    return jsonify(
        round=round_id,
        client_id=current_app.config["CLIENT_ID"],
        malicious=malicious,
        valid=True,
        update={
            "model_delta": -3.5 if attacked else model_delta,
            "scale": -5.0 if attacked else 1.0,
        },
    )


@experiment_bp.get("/api/training/status")
def get_training_status():
    """현재 클라이언트의 최신 로컬 학습 결과를 반환한다."""
    return jsonify(
        client_id=current_app.config["CLIENT_ID"],
        training=current_app.extensions.get("latest_local_training", {}),
    )


@experiment_bp.get("/api/training/history")
def get_training_history():
    """라운드별 로컬 학습 결과 기록을 반환한다."""
    return jsonify(
        client_id=current_app.config["CLIENT_ID"],
        count=len(current_app.extensions.get("local_training_results", [])),
        results=current_app.extensions.get("local_training_results", []),
    )
