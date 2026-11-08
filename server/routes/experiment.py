from flask import Blueprint, current_app, jsonify, request
from config.model_state import create_random_global_model


experiment_bp = Blueprint("server_experiment", __name__)


@experiment_bp.post("/api/experiment/config")
def set_config():
    # 중앙 서버의 라운드 집계 및 방어 동작에 사용할 설정을 저장한다.
    current_app.extensions["experiment_config"] = request.get_json(silent=True) or {}
    return jsonify(status="configured", config=current_app.extensions["experiment_config"])


@experiment_bp.get("/api/experiment/config")
def get_config():
    return jsonify(current_app.extensions["experiment_config"])


@experiment_bp.post("/api/round/config")
def set_round_config():
    payload = request.get_json(silent=True) or {}
    current_app.extensions["round_config"] = payload
    if current_app.extensions.get("global_model") is None:
        current_app.extensions["global_model"] = create_random_global_model(
            seed=payload.get("seed", 42),
            num_parameters=payload.get("num_parameters", 32),
        )
    return jsonify(status="configured", round=payload.get("round"), config=payload)


@experiment_bp.get("/api/round/config")
def get_round_config():
    return jsonify(current_app.extensions.get("round_config", {}))


@experiment_bp.get("/api/model/global")
def get_global_model():
    return jsonify(current_app.extensions.get("global_model", {}))


@experiment_bp.get("/api/model/history")
def get_global_model_history():
    """라운드별 FedAvg 전역 모델 갱신 기록을 반환한다."""
    history = current_app.extensions.get("global_model_history", [])
    return jsonify(count=len(history), history=history)
