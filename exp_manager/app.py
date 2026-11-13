import argparse
import atexit
import sys
from pathlib import Path

PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from flask import Flask
from werkzeug.serving import make_server

from exp_manager.routes.api import api_bp
from exp_manager.services.experiment_runner import ExperimentManager
from exp_manager.services.process_manager import ProcessManager
from exp_manager.services.data_distribution import DataDistributionManager
from exp_manager.utils.project_root_path import PROJECT_ROOT_PATH


def create_app(
    host: str, port: int, client_count: int, server_port: int, client_base_port: int
):
    app = Flask(__name__)
    app.extensions["server"] = None

    process_manager = ProcessManager(
        host, client_count, server_port, client_base_port, port
    )
    data_manager = DataDistributionManager(process_manager, client_count)
    experiment_manager = ExperimentManager(
        process_manager=process_manager,
        results_dir=PROJECT_ROOT_PATH / "results",
        client_count=client_count,
    )
    experiment_manager.data_manager = data_manager
    app.extensions["process_manager"] = process_manager
    app.extensions["experiment_manager"] = experiment_manager
    app.extensions["data_manager"] = data_manager

    app.register_blueprint(api_bp)

    atexit.register(process_manager.stop)
    return app


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--client-count", type=int, default=10)
    parser.add_argument("--server-port", type=int, default=5000)
    parser.add_argument("--client-base-port", type=int, default=5101)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    flask_app = create_app(
        args.host,
        args.port,
        args.client_count,
        args.server_port,
        args.client_base_port,
    )
    flask_app.extensions["process_manager"].start()
    http_server = make_server(args.host, args.port, flask_app, threaded=True)
    flask_app.extensions["server"] = http_server

    print(
        f"Experiment manager listening on http://{args.host}:{args.port}",
        flush=True,
    )
    http_server.serve_forever()
