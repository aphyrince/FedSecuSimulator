from flask import Blueprint, current_app, jsonify, request


experiment_bp = Blueprint("experiment", __name__)


@experiment_bp.get("/api/experiment/config")
def get_config():
    return jsonify(current_app.extensions["experiment_manager"].get_public_config())


@experiment_bp.post("/api/experiment/config")
def set_config():
    """웹 UI의 실험 옵션을 검증한 뒤 실행 서비스에 저장한다."""
    try:
        manager = current_app.extensions["experiment_manager"]
        manager.update_config(request.get_json(silent=True) or {})
        return jsonify(manager.get_public_config())
    except (ValueError, TypeError) as exc:
        return jsonify(error=str(exc)), 400


@experiment_bp.post("/api/experiment/start")
def start():
    # 가상 라운드 실행을 백그라운드 스레드에서 시작한다.
    try:
        return jsonify(current_app.extensions["experiment_manager"].start())
    except (RuntimeError, ValueError, TypeError, OSError) as exc:
        return jsonify(error=str(exc)), 409


@experiment_bp.post("/api/experiment/continuous/start")
def start_continuous():
    """24개 내부자 협력공격 실험을 순차 실행한다."""
    try:
        payload = request.get_json(silent=True) or {}
        if not isinstance(payload, dict):
            return jsonify(error="연속 실험 설정은 객체 형식이어야 합니다."), 400
        cases = payload.get("cases")
        return jsonify(
            current_app.extensions["experiment_manager"].start_continuous_experiments(
                cases
            )
        )
    except (RuntimeError, ValueError, TypeError) as exc:
        return jsonify(error=str(exc)), 409


@experiment_bp.post("/api/experiment/continuous/dlg/start")
def start_continuous_dlg():
    """사용자 체크포인트와 고정 이미지를 사용하는 DLG 3개 실험을 시작한다."""
    try:
        payload = request.get_json(silent=True) or {}
        return jsonify(
            current_app.extensions[
                "experiment_manager"
            ].start_continuous_dlg_experiments(payload)
        )
    except (RuntimeError, ValueError, TypeError) as exc:
        return jsonify(error=str(exc)), 409


@experiment_bp.post("/api/experiment/continuous/linked/start")
def start_continuous_linked():
    """Run all federated cases and the three DLG noise cases per checkpoint."""
    try:
        payload = request.get_json(silent=True) or {}
        if not isinstance(payload, dict):
            return jsonify(error="연결 실험 설정은 객체 형식이어야 합니다."), 400
        return jsonify(
            current_app.extensions[
                "experiment_manager"
            ].start_continuous_linked_experiments(
                payload.get("cases"), payload.get("dlg", {})
            )
        )
    except (RuntimeError, ValueError, TypeError, OSError) as exc:
        return jsonify(error=str(exc)), 409


@experiment_bp.post("/api/experiment/stop")
def stop():
    return jsonify(current_app.extensions["experiment_manager"].stop())


@experiment_bp.get("/api/experiment/status")
def status():
    return jsonify(current_app.extensions["experiment_manager"].status())


@experiment_bp.get("/api/results/continuous")
def continuous_results():
    return jsonify(
        current_app.extensions[
            "experiment_manager"
        ].continuous_results_summary()
    )


@experiment_bp.post("/api/experiment/reset")
def reset():
    try:
        return jsonify(current_app.extensions["experiment_manager"].reset())
    except RuntimeError as exc:
        return jsonify(error=str(exc)), 409


@experiment_bp.post("/api/experiment/replay")
def replay():
    """저장된 연합학습 라운드를 설정·데이터·체크포인트 스냅샷으로 재실행한다."""
    payload = request.get_json(silent=True) or {}
    try:
        return jsonify(
            current_app.extensions["experiment_manager"].replay_round(
                payload["experiment_id"], payload["round"]
            )
        )
    except (KeyError, TypeError, ValueError, RuntimeError, OSError) as exc:
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


@experiment_bp.post("/api/dlg/original")
def receive_dlg_original():
    """신뢰된 평가 관리자만 단일 이미지 원본을 보관한다. 서버 복원에는 전달하지 않는다."""
    payload = request.get_json(silent=True) or {}
    manager = current_app.extensions["experiment_manager"]
    try:
        manager.store_dlg_original(
            round_id=int(payload["round"]),
            experiment_id=payload["experiment_id"],
            client_id=int(payload["client_id"]),
            image=payload["image"],
            label=payload.get("label"),
        )
    except (KeyError, TypeError, ValueError) as exc:
        return jsonify(error=str(exc)), 400
    return jsonify(status="original_saved", round=payload["round"])


@experiment_bp.post("/api/dlg/results")
def receive_dlg_result():
    """서버 복원 결과와 원본 이미지로 DLG MSE를 계산해 기록한다."""
    result = request.get_json(silent=True) or {}
    round_id = int(result.get("round", 0))
    manager = current_app.extensions["experiment_manager"]
    original = manager.get_dlg_original(round_id)
    restored = result.get("reconstructed_image", [])
    if len(original) != len(restored):
        return jsonify(
            error="Original and reconstructed image dimensions do not match",
            original_size=len(original),
            reconstructed_size=len(restored),
        ), 400
    mse = sum(
        (float(original[index]) - float(restored[index])) ** 2
        for index in range(len(original))
    ) / max(1, len(original))
    result["original_image"] = original
    result["mse"] = round(mse, 8)
    result["round_type"] = "dlg"
    result["experiment_type"] = "dlg"
    result["image_shape"] = result.get("image_shape", [len(original)])
    result["defense_enabled"] = bool(result.get("defense_enabled", False))
    result["noise_sigma"] = (
        float(result.get("noise_sigma", 0.0))
        if result["defense_enabled"]
        else 0.0
    )
    manager.record_dlg_result(round_id, result)
    return jsonify(status="received", round=round_id, mse=result["mse"])
