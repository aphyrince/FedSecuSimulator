from flask import Blueprint

health_bp = Blueprint("health", __name__, url_prefix="api/admin/health")


@health_bp.route("/", methods=["GET"])
def get_exp_manager_health():
    return "experiment manager's health is good."


@health_bp.route("/server", methods=["GET"])
def get_server_health():
    return "server health is good."


@health_bp.route("/clients", methods=["GET"])
def get_clients_health():
    return "All client's health is good. "


@health_bp.route("/client/<int:client_id>", methods=["GET"])
def get_client_health_by_id(client_id):
    return "client_" + client_id + "'s health is good."
