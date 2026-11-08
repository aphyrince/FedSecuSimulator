from flask import Blueprint, current_app, jsonify, request

from utils.shutdown import schedule_shutdown


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
    return jsonify(
        client_id=current_app.config["CLIENT_ID"],
        port=current_app.config["PORT"],
        available=current_app.extensions["client_state"].available,
        status="running",
    )


@system_bp.post("/api/availability")
def availability():
    """현재 라운드 참여 가능 여부를 중앙 서버의 질의에 응답한다."""
    app = current_app._get_current_object()
    available = app.extensions["client_state"].available
    payload = request.get_json(silent=True) or {}
    return jsonify(
        client_id=app.config["CLIENT_ID"],
        round=payload.get("round"),
        available=bool(available),
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
    app.extensions["client_state"].expired_rounds = (
        app.extensions["client_state"].expired_rounds | {int(payload.get("round", 0))}
    )
    return jsonify(
        status="expired",
        client_id=app.config["CLIENT_ID"],
        round=payload.get("round"),
    )
