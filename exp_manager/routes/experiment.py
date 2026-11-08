from flask import Blueprint, current_app, jsonify, request


experiment_bp = Blueprint("experiment", __name__)


@experiment_bp.get("/api/experiment/config")
def get_config():
    return jsonify(current_app.extensions["experiment_manager"].get_config())


@experiment_bp.post("/api/experiment/config")
def set_config():
    """웹 UI의 실험 옵션을 검증한 뒤 실행 서비스에 저장한다."""
    try:
        manager = current_app.extensions["experiment_manager"]
        return jsonify(manager.update_config(request.get_json(silent=True) or {}))
    except (ValueError, TypeError) as exc:
        return jsonify(error=str(exc)), 400


@experiment_bp.post("/api/experiment/start")
def start():
    # 가상 라운드 실행을 백그라운드 스레드에서 시작한다.
    try:
        return jsonify(current_app.extensions["experiment_manager"].start())
    except RuntimeError as exc:
        return jsonify(error=str(exc)), 409


@experiment_bp.post("/api/experiment/stop")
def stop():
    return jsonify(current_app.extensions["experiment_manager"].stop())


@experiment_bp.get("/api/experiment/status")
def status():
    return jsonify(current_app.extensions["experiment_manager"].status())


@experiment_bp.post("/api/experiment/reset")
def reset():
    try:
        return jsonify(current_app.extensions["experiment_manager"].reset())
    except RuntimeError as exc:
        return jsonify(error=str(exc)), 409


@experiment_bp.post("/api/rounds/<int:round_id>/execute")
def execute_round(round_id):
    """단일 라운드 설정을 실행하고 중앙 서버 결과를 관리자 API로 반환한다."""
    try:
        result = current_app.extensions["experiment_manager"].execute_single_round(
            round_id, request.get_json(silent=True) or {}
        )
        return jsonify(result)
    except (RuntimeError, ValueError, OSError) as exc:
        return jsonify(error=str(exc)), 409


@experiment_bp.post("/api/results/round")
def receive_round_result():
    """중앙 서버로부터 라운드 결과를 수신해 관리자 상태에 저장한다."""
    result = request.get_json(silent=True) or {}
    round_id = int(result.get("round", 0))
    if round_id < 1:
        return jsonify(error="Round number must be positive"), 400
    manager = current_app.extensions["experiment_manager"]
    round_path = manager.record_round_result(result)
    return jsonify(
        status="received",
        round=round_id,
        recorded_file=str(round_path),
    )


@experiment_bp.post("/api/dlg/results")
def receive_dlg_result():
    """서버 복원 결과와 원본 이미지로 DLG MSE를 계산해 기록한다."""
    result = request.get_json(silent=True) or {}
    round_id = int(result.get("round", 0))
    manager = current_app.extensions["experiment_manager"]
    original = manager.get_dlg_original(round_id)
    restored = result.get("reconstructed_image", [])
    mse = sum(
        (float(original[index]) - float(restored[index])) ** 2
        for index in range(min(len(original), len(restored)))
    ) / max(1, len(restored))
    result["original_image"] = original
    result["mse"] = round(mse, 8)
    manager.record_dlg_result(round_id, result)
    return jsonify(status="received", round=round_id, mse=result["mse"])
