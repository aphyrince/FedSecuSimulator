from flask import Blueprint

from clients.routes.dataset import dataset_bp
from clients.routes.experiment import experiment_bp
from clients.routes.system import system_bp


# 각 클라이언트의 시스템, 데이터, 실험 API를 기능별로 구성한다.
api_bp = Blueprint("client_api", __name__)
api_bp.register_blueprint(system_bp)
api_bp.register_blueprint(dataset_bp)
api_bp.register_blueprint(experiment_bp)
