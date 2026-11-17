from flask import Blueprint

operation_bp = Blueprint("operation", __name__, url_prefix="api/admin/operation")


@operation_bp.route("/exp/stop", methods=["GET"])
def exp_manager_stop():
    return "exp_stop?"


@operation_bp.route("/server/start", methods=["GET"])
def central_server_start():
    return "central_server_start"


@operation_bp.route("/server/stop", methods=["GET"])
def central_server_stop():
    return "central_server_stop"


@operation_bp.route("/clients/start", methods=["GET"])
def clients_start():
    return "clients_start"


@operation_bp.route("/clients/stop", methods=["GET"])
def clients_stop():
    return "clients_stop"


@operation_bp.route("/client/<int:client_id>/start", methods=["GET"])
def client_start_by_id(client_id):
    return "client_start_" + client_id


@operation_bp.route("/client/<int:client_id>/stop", methods=["GET"])
def client_stop_by_id(client_id):
    return "client_stop_" + client_id
