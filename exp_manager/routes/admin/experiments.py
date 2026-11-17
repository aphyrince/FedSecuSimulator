from flask import Blueprint

experiments_bp = Blueprint("experiments", __name__, url_prefix="api/admin/experiments")


# 실험 설정 조회
@experiments_bp.route("/config", methods=["GET"])
def get_experiments_config():
    return "get_experiments_config"


# 실험 설정 변경
@experiments_bp.route("/config", methods=["POST"])
def set_experiments_config():
    return "set_experiments_config"


# 실험 시작
@experiments_bp.route("/start", methods=["GET"])
def start_experiments():
    return "start_experiments"


# 실험 중단
@experiments_bp.route("/stop", methods=["GET"])
def stop_experiments():
    return "stop_experiments"


# 실험 진행상황 조회
@experiments_bp.route("/status", methods=["GET"])
def get_experiments_status():
    return "get_experiments_status"


# 실험 초기화
@experiments_bp.route("/reset", methods=["GET"])
def reset_experiments():
    return "reset_experiments"
