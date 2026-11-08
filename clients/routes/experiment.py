from flask import Blueprint, current_app, jsonify, request


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
    return jsonify(
        status="received",
        client_id=current_app.config["CLIENT_ID"],
        model_version=payload.get("model_version"),
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
    return jsonify(
        round=payload.get("round", 1),
        client_id=current_app.config["CLIENT_ID"],
        malicious=malicious,
        valid=True,
        update={
            "model_delta": -3.5 if attacked else 0.82,
            "scale": -5.0 if attacked else 1.0,
        },
    )
