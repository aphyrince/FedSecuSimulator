from flask import Blueprint, current_app, jsonify, request


dataset_bp = Blueprint("server_dataset", __name__)


@dataset_bp.post("/api/dataset/range")
def set_range():
    """실험 관리자에게 받은 테스트 데이터 파일과 인덱스 범위를 저장한다."""
    current_app.extensions["dataset_range"] = request.get_json(silent=True) or {}
    return jsonify(status="configured", dataset=current_app.extensions["dataset_range"])


@dataset_bp.get("/api/dataset/range")
def get_range():
    return jsonify(current_app.extensions.get("dataset_range", {}))
