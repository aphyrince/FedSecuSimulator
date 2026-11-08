from flask import Blueprint, current_app, jsonify, request

from services.training_service import build_round_update, run_local_training
from services.server_client import post_json

experiment_bp = Blueprint("client_experiment", __name__)


@experiment_bp.post("/api/experiment/config")
def set_config():
    # 실험에서 사용할 라운드·참여·공격·방어 설정을 저장한다.
    app = current_app._get_current_object()
    state = app.extensions["client_state"]
    config = state.set_experiment_config(request.get_json(silent=True) or {})
    return jsonify(
        status="configured",
        client_id=app.config["CLIENT_ID"],
        config=config,
    )


@experiment_bp.get("/api/experiment/config")
def get_config():
    return jsonify(current_app.extensions["client_state"].experiment_config)


@experiment_bp.post("/api/model/global")
def receive_global_model():
    """중앙 서버가 전달한 전역 모델을 수신해 로컬 최신 모델로 저장한다."""
    payload = request.get_json(silent=True) or {}
    state = current_app.extensions["client_state"]
    state.receive_global_model(payload)

    training_record = run_local_training(
        state=state,
        client_id=current_app.config["CLIENT_ID"],
        global_model=payload,
    )

    return jsonify(
        status="received",
        client_id=current_app.config["CLIENT_ID"],
        model_version=payload.get("model_version"),
        training=training_record,
    )


@experiment_bp.get("/api/model/global")
def get_global_model():
    return jsonify(current_app.extensions["client_state"].global_model)


@experiment_bp.post("/api/round/execute")
def execute_round():
    """실험 관리자가 요청한 로컬 라운드 업데이트를 생성한다."""
    payload = request.get_json(silent=True) or {}
    update = build_round_update(
        state=current_app.extensions["client_state"],
        client_id=current_app.config["CLIENT_ID"],
        request_payload=payload,
    )
    return jsonify(update)


@experiment_bp.post("/api/round/submit")
def submit_round_update():
    """추출한 로컬 업데이트를 중앙 서버에 직접 제출한다."""
    payload = request.get_json(silent=True) or {}
    round_id = int(payload.get("round", 1))
    state = current_app.extensions["client_state"]
    update = build_round_update(
        state=state,
        client_id=current_app.config["CLIENT_ID"],
        request_payload=payload,
    )
    server_url = payload.get("server_url", current_app.config["SERVER_URL"])
    response = post_json(
        f"{server_url.rstrip('/')}/api/rounds/{round_id}/updates",
        update,
    )
    state.save_submission(round_id, response)
    return jsonify(
        status="submitted",
        client_id=current_app.config["CLIENT_ID"],
        round=round_id,
        server_response=response,
    )


@experiment_bp.get("/api/training/status")
def get_training_status():
    """현재 클라이언트의 최신 로컬 학습 결과를 반환한다."""
    return jsonify(
        client_id=current_app.config["CLIENT_ID"],
        training=current_app.extensions["client_state"].latest_local_training,
    )


@experiment_bp.get("/api/training/history")
def get_training_history():
    """라운드별 로컬 학습 결과 기록을 반환한다."""
    state = current_app.extensions["client_state"]
    return jsonify(
        client_id=current_app.config["CLIENT_ID"],
        count=len(state.local_training_results),
        results=state.local_training_results,
    )


@experiment_bp.get("/api/updates/extracted/<int:round_id>")
def get_extracted_update(round_id):
    """학습 전후 모델 차이로 추출한 라운드 업데이트를 반환한다."""
    return jsonify(
        client_id=current_app.config["CLIENT_ID"],
        round=round_id,
        update=current_app.extensions["client_state"].extracted_updates.get(round_id),
    )


@experiment_bp.get("/api/updates/submitted/<int:round_id>")
def get_submitted_update(round_id):
    """중앙 서버에 제출한 라운드 업데이트 응답을 반환한다."""
    return jsonify(
        client_id=current_app.config["CLIENT_ID"],
        round=round_id,
        submission=current_app.extensions["client_state"].submitted_updates.get(round_id),
    )
