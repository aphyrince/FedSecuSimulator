from flask import Blueprint, current_app, jsonify, request


dataset_bp = Blueprint("client_dataset", __name__)


@dataset_bp.post("/api/dataset/range")
def set_range():
    """실험 관리자에게 받은 훈련 데이터 범위를 클라이언트 상태에 저장한다."""
    app = current_app._get_current_object()
    state = app.extensions["client_state"]
    dataset_range = state.set_dataset_range(request.get_json(silent=True) or {})
    return jsonify(
        status="configured",
        client_id=app.config["CLIENT_ID"],
        dataset=dataset_range,
    )


@dataset_bp.get("/api/dataset/range")
def get_range():
    state = current_app.extensions["client_state"]
    return jsonify(state.dataset_range)
