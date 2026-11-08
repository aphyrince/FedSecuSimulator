from flask import Blueprint

from exp_manager.routes.data import data_bp
from exp_manager.routes.experiment import experiment_bp
from exp_manager.routes.system import system_bp


# 기능별 Blueprint를 하나의 관리자 API로 등록한다.
api_bp = Blueprint("api", __name__)
api_bp.register_blueprint(system_bp)
api_bp.register_blueprint(experiment_bp)
api_bp.register_blueprint(data_bp)
