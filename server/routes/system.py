from flask import Blueprint, current_app, jsonify, request

from utils.shutdown import schedule_shutdown


system_bp = Blueprint("server_system", __name__)


@system_bp.get("/health")
def health():
    """중앙 서버 Flask 프로세스의 실행 상태를 확인한다."""
    return jsonify(
        status="running",
        role="server",
        host=current_app.config["HOST"],
        port=current_app.config["PORT"],
    )


@system_bp.get("/api/status")
def status():
    # 라운드 집계 결과 수를 포함한 중앙 서버 상태를 반환한다.
    return jsonify(
        status="running",
        host=current_app.config["HOST"],
        port=current_app.config["PORT"],
        clients_registered=len(current_app.extensions.get("clients", {})),
        rounds_completed=len(current_app.extensions["round_results"]),
    )


@system_bp.post("/api/clients/register")
def register_clients():
    """실험 관리자가 생성한 클라이언트 id와 주소 정보를 중앙 서버에 저장한다."""
    payload = request.get_json(silent=True) or {}
    clients = payload.get("clients", [])
    current_app.extensions["clients"] = {
        client["client_id"]: client for client in clients if "client_id" in client
    }
    return jsonify(status="registered", count=len(clients), clients=clients)


@system_bp.get("/api/clients")
def registered_clients():
    clients = current_app.extensions.get("clients", {})
    return jsonify(clients=list(clients.values()), count=len(clients))


@system_bp.get("/api/rounds/<int:round_id>/availability")
def round_availability(round_id):
    """특정 라운드의 참여 가능 여부 질의 응답과 참여 가능 클라이언트를 반환한다."""
    responses = current_app.extensions.get("availability_responses", {}).get(round_id, [])
    available = current_app.extensions.get("available_clients", {}).get(round_id, [])
    return jsonify(
        round=round_id,
        responses=responses,
        available_clients=available,
    )


@system_bp.post("/api/shutdown")
def shutdown():
    # 실험 관리자 종료 요청을 받은 경우 Flask 서버를 스케줄링하여 종료한다.
    schedule_shutdown(current_app._get_current_object())
    return jsonify(message="Shutdown scheduled", role="server")
