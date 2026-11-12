from flask import Blueprint, current_app, jsonify, request

from clients.utils.shutdown import schedule_shutdown
from clients.services.local_training import count_dataset_samples


system_bp = Blueprint("client_system", __name__)


@system_bp.get("/health")
def health():
    """개별 클라이언트 Flask 프로세스의 실행 상태를 확인한다."""
    return jsonify(
        status="running",
        role="client",
        client_id=current_app.config["CLIENT_ID"],
        port=current_app.config["PORT"],
    )


@system_bp.get("/api/status")
def status():
    state = current_app.extensions["client_state"]
    assigned_samples = count_dataset_samples(state.dataset_range)
    training = dict(state.training_progress)
    model_training = state.global_model.get("training_config", {})
    max_samples = model_training.get("max_train_samples")
    trainable_samples = (
        min(assigned_samples, max_samples)
        if max_samples is not None
        else assigned_samples
    )
    round_context = state.global_model.get("round_context", {})
    available = (
        state.available
        and bool(state.dataset_range.get("ranges"))
        and not state.training_lock.locked()
    )
    return jsonify(
        client_id=current_app.config["CLIENT_ID"],
        port=current_app.config["PORT"],
        available=available,
        status="running",
        dataset_samples=assigned_samples,
        trainable_samples=trainable_samples,
        training=training,
        training_lock=state.training_lock.locked(),
        current_round=state.global_model.get("round"),
        selected=bool(round_context.get("selected", False)),
        malicious=bool(round_context.get("malicious", False)),
        delayed=bool(round_context.get("delayed", False)),
    )


@system_bp.post("/api/availability")
def availability():
    """현재 라운드 참여 가능 여부를 중앙 서버의 질의에 응답한다."""
    app = current_app._get_current_object()
    available = app.extensions["client_state"].available
    state = app.extensions["client_state"]
    available = available and bool(state.dataset_range.get("ranges")) and not state.training_lock.locked()
    payload = request.get_json(silent=True) or {}
    return jsonify(
        client_id=app.config["CLIENT_ID"],
        round=payload.get("round"),
        available=bool(available),
        num_samples=sum(item["end"] - item["start"] for item in state.dataset_range.get("ranges", [])),
    )


@system_bp.post("/api/shutdown")
def shutdown():
    # 실험 관리자 종료 요청을 받아 현재 클라이언트 서버만 종료한다.
    app = current_app._get_current_object()
    schedule_shutdown(app, app.config["CLIENT_ID"])
    return jsonify(message="Shutdown scheduled", client_id=app.config["CLIENT_ID"])


@system_bp.post("/api/round/timeout")
def round_timeout():
    """중앙 서버가 설정된 제출 기한을 초과했음을 알린다."""
    payload = request.get_json(silent=True) or {}
    app = current_app._get_current_object()
    state = app.extensions["client_state"]
    round_id = int(payload.get("round", 0))
    state.expired_rounds = state.expired_rounds | {round_id}
    if state.training_progress.get("round") == round_id:
        state.training_progress["status"] = "expired"
    return jsonify(
        status="expired",
        client_id=app.config["CLIENT_ID"],
        round=round_id,
    )
