import argparse
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from flask import Flask
from werkzeug.serving import make_server

from routes.api import api_bp


def create_app(host: str, port: int):
    app = Flask(__name__)
    app.config["HOST"] = host
    app.config["PORT"] = port
    app.extensions["server"] = None
    app.extensions["experiment_config"] = {}
    app.extensions["round_results"] = []
    app.extensions["dataset_range"] = {}
    app.extensions["clients"] = {}
    app.extensions["round_config"] = {}
    app.extensions["available_clients"] = {}
    app.extensions["availability_responses"] = {}
    app.extensions["selected_clients"] = {}
    app.extensions["round_updates"] = {}
    app.extensions["update_submission_status"] = {}
    app.extensions["global_model"] = None
    app.extensions["global_model_history"] = []
    app.extensions["evaluation_results"] = {}

    app.register_blueprint(api_bp)
    return app


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    flask_app = create_app(args.host, args.port)
    http_server = make_server(args.host, args.port, flask_app, threaded=True)
    flask_app.extensions["server"] = http_server

    print(f"Central server listening on http://{args.host}:{args.port}", flush=True)
    http_server.serve_forever()
