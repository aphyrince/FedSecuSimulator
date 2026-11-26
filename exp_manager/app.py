import argparse
import atexit
import signal
import sys
import threading
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
from common.runtime_log import configure_runtime_logging


def _register_sigterm_handler(process_manager, http_server):
    shutdown_requested = threading.Event()

    def handle_sigterm(_signum, _frame):
        if shutdown_requested.is_set():
            return
        shutdown_requested.set()

        def stop_children_and_server():
            process_manager.stop()
            http_server.shutdown()

        threading.Thread(
            target=stop_children_and_server,
            name="process-manager-shutdown",
            daemon=True,
        ).start()

    signal.signal(signal.SIGTERM, handle_sigterm)


def create_app(
    host: str, port: int, client_count: int, server_port: int, client_base_port: int,
    *, sequential_clients: bool = False, e32_train_samples_per_client: int | None = None,
):
    app = Flask(__name__)
    app.extensions["server"] = None

    process_manager = ProcessManager(
        host, client_count, server_port, client_base_port, port,
        sequential_clients=sequential_clients,
    )
    data_manager = DataDistributionManager(
        process_manager, client_count, e32_train_samples_per_client=e32_train_samples_per_client,
    )
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
    parser.add_argument("--sequential-clients", action="store_true", help="Run normal FedAvg clients in the manager process")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    configure_runtime_logging("manager")
    flask_app = create_app(
        args.host,
        args.port,
        args.client_count,
        args.server_port,
        args.client_base_port,
        sequential_clients=args.sequential_clients,
    )
    flask_app.extensions["process_manager"].start()
    http_server = make_server(args.host, args.port, flask_app, threaded=True)
    flask_app.extensions["server"] = http_server

    print(
        f"Experiment manager listening on http://{args.host}:{args.port}",
        flush=True,
    )
    _register_sigterm_handler(
        flask_app.extensions["process_manager"], http_server
    )
    http_server.serve_forever()
