from flask import Blueprint

from server.routes.aggregation import aggregation_bp
from server.routes.dataset import dataset_bp
from server.routes.experiment import experiment_bp
from server.routes.system import system_bp
from server.routes.dlg import dlg_bp


# 중앙 서버의 시스템·실험·데이터·집계 API를 기능별로 구성한다.
api_bp = Blueprint("server_api", __name__)
api_bp.register_blueprint(system_bp)
api_bp.register_blueprint(dataset_bp)
api_bp.register_blueprint(experiment_bp)
api_bp.register_blueprint(aggregation_bp)
api_bp.register_blueprint(dlg_bp)
