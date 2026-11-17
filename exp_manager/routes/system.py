import threading

from flask import Blueprint, current_app, jsonify


system_bp = Blueprint("experiment_system", __name__)


@system_bp.get("/")
def index():
    from flask import render_template

    manager = current_app.extensions["process_manager"]
    return render_template("manager.html", status=manager.status())


@system_bp.get("/health")
def health():
    return jsonify(status="running", role="experiment-manager")


@system_bp.get("/api/status")
def status():
    """실행 중인 중앙 서버와 각 클라이언트 프로세스 상태를 조회한다."""
    return jsonify(current_app.extensions["process_manager"].status())


@system_bp.post("/api/start")
def start():
    # 중앙 서버와 클라이언트 프로세스를 모두 생성한다.
    return jsonify(current_app.extensions["process_manager"].start())


@system_bp.post("/api/stop")
def stop():
    # 자식 Flask 프로세스만 종료하고 실험 관리자는 유지한다.
    manager = current_app.extensions["process_manager"]
    manager.stop()
    return jsonify(message="All processes stopped", **manager.status())


@system_bp.post("/api/shutdown")
def shutdown():
    """자식 프로세스를 REST API로 종료한 뒤 실험 관리자 자체를 종료한다."""
    current_app.extensions["process_manager"].stop()
    http_server = current_app.extensions["server"]
    if http_server is not None:
        threading.Thread(target=http_server.shutdown, daemon=True).start()
    return jsonify(message="Experiment manager shutdown scheduled")
