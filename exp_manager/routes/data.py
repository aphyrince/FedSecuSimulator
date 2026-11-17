from flask import Blueprint, current_app, jsonify, request


data_bp = Blueprint("experiment_data", __name__)


@data_bp.get("/api/data/status")
def status():
    """현재 .env 설정과 서버/클라이언트별 데이터 범위 상태를 반환한다."""
    return jsonify(current_app.extensions["data_manager"].status())


@data_bp.post("/api/data/distribute")
def distribute():
    """훈련 데이터 범위와 테스트 데이터 범위를 자식 서버에 분배한다."""
    payload = request.get_json(silent=True) or {}
    try:
        manager = current_app.extensions["data_manager"]
        return jsonify(
            manager.distribute(
                payload.get("seed", 42),
                payload.get("distribution", "iid"),
            )
        )
    except (FileNotFoundError, ValueError, OSError, TimeoutError) as exc:
        return jsonify(error=str(exc)), 400
