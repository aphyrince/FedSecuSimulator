from flask import Blueprint, current_app, jsonify, request

from server.utils.shutdown import schedule_shutdown


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
    extensions = current_app.extensions
    progress = dict(extensions.get("round_progress", {}))
    round_id = progress.get("round")
    clients = list(extensions.get("clients", {}).values())
    available = extensions.get("available_clients", {}).get(round_id, [])
    selected = extensions.get("selected_clients", {}).get(round_id, [])
    config = progress.get("config") or extensions.get("round_config", {})
    malicious = (
        set(config.get("malicious_clients", []))
        if config.get("attack_enabled", False)
        else set()
    )
    delayed = set(config.get("delay_clients", []))
    responses = {
        item["client_id"]: item
        for item in extensions.get("availability_responses", {}).get(round_id, [])
    }
    update_status = extensions.get("update_submission_status", {}).get(round_id, {})
    updates = extensions.get("round_updates", {}).get(round_id, {})
    client_activity = [
        {
            "client_id": client["client_id"],
            "available": client["client_id"] in available,
            "selected": client["client_id"] in selected,
            "malicious": client["client_id"] in malicious and client["client_id"] in selected,
            "delayed": client["client_id"] in delayed and client["client_id"] in selected,
            "availability_samples": responses.get(client["client_id"], {}).get("num_samples"),
            "expected_samples": extensions.get("expected_sample_counts", {})
            .get(round_id, {})
            .get(client["client_id"]),
            "update_status": update_status.get(client["client_id"], {}).get(
                "status", "not_selected" if client["client_id"] not in selected else "waiting"
            ),
            "submitted_samples": updates.get(client["client_id"], {}).get("num_samples"),
        }
        for client in clients
    ]
    results = extensions.get("round_results", [])
    last_result = results[-1] if results else {}
    return jsonify(
        status="running",
        host=current_app.config["HOST"],
        port=current_app.config["PORT"],
        clients_registered=len(clients),
        rounds_completed=len(results),
        current_round=round_id,
        round_phase=progress.get("phase", "idle"),
        round_message=progress.get("message", ""),
        round_active=bool(extensions.get("round_lock") and extensions["round_lock"].locked()),
        available_count=len(available),
        selected_count=len(selected),
        updates_received=len(updates),
        malicious_clients=sorted(malicious.intersection(selected)),
        delayed_clients=sorted(delayed.intersection(selected)),
        participation_rate=config.get("participation_rate"),
        attack_enabled=bool(config.get("attack_enabled", False)),
        defense_enabled=bool(config.get("defense_enabled", False)),
        defense=config.get("defense", "none"),
        max_train_samples=config.get("max_train_samples"),
        update_deadline=extensions.get("update_deadlines", {}).get(round_id),
        global_model_version=(extensions.get("global_model") or {}).get("model_version"),
        last_accuracy=last_result.get("accuracy"),
        last_loss=last_result.get("loss"),
        client_activity=client_activity,
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


@system_bp.get("/api/rounds/<int:round_id>/selected-clients")
def round_selected_clients(round_id):
    """라운드 참여자로 선정된 클라이언트 ID 목록을 반환한다."""
    selected = current_app.extensions.get("selected_clients", {}).get(round_id, [])
    return jsonify(round=round_id, selected_clients=selected, count=len(selected))


@system_bp.post("/api/shutdown")
def shutdown():
    # 실험 관리자 종료 요청을 받은 경우 Flask 서버를 스케줄링하여 종료한다.
    schedule_shutdown(current_app._get_current_object())
    return jsonify(message="Shutdown scheduled", role="server")
