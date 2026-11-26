from flask import Blueprint, current_app, jsonify, request


dataset_bp = Blueprint("server_dataset", __name__)


@dataset_bp.post("/api/dataset/range")
def set_range():
    """실험 관리자에게 받은 테스트 데이터 파일과 인덱스 범위를 저장한다."""
    payload = request.get_json(silent=True) or {}
    if not isinstance(payload, dict):
        return jsonify(error="Dataset range must be an object"), 400
    final_test = payload.pop("final_test_range", None)
    if final_test is not None and (
        not isinstance(final_test, dict) or not final_test.get("ranges")
        or final_test.get("split") != "test" or payload.get("split") != "validation"
    ):
        return jsonify(error="Expected validation and final test dataset ranges"), 400
    state = current_app.extensions
    with state["final_test_lock"]:
        state["dataset_range"] = payload
        if state["final_test_range"] != final_test:
            state["final_test_range"] = final_test
            state["final_test_evaluation"] = None
    return jsonify(status="configured", dataset=payload)


@dataset_bp.get("/api/dataset/range")
def get_range():
    return jsonify(current_app.extensions.get("dataset_range", {}))
