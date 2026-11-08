from flask import Blueprint

from routes.aggregation import aggregation_bp
from routes.dataset import dataset_bp
from routes.experiment import experiment_bp
from routes.system import system_bp


# 중앙 서버의 시스템·실험·데이터·집계 API를 기능별로 구성한다.
api_bp = Blueprint("server_api", __name__)
api_bp.register_blueprint(system_bp)
api_bp.register_blueprint(dataset_bp)
api_bp.register_blueprint(experiment_bp)
api_bp.register_blueprint(aggregation_bp)
