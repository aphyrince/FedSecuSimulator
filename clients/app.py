import argparse
import os
import sys

from flask import Flask
from werkzeug.serving import make_server

from routes.api import api_bp


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def create_app(client_id: int, port: int):
    app = Flask(__name__)
    app.config["CLIENT_ID"] = client_id
    app.config["PORT"] = port
    app.extensions["server"] = None
    app.extensions["experiment_config"] = {}
    app.extensions["dataset_range"] = {}
    app.extensions["available"] = True
    app.extensions["global_models"] = []
    app.extensions["global_model"] = {}
    app.extensions["local_training_results"] = []
    app.extensions["latest_local_training"] = {}
    app.register_blueprint(api_bp)
    return app


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--client-id", required=True, type=int)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", required=True, type=int)
    parser.add_argument("--server-url", default="http://127.0.0.1:5000")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    flask_app = create_app(args.client_id, args.port)
    http_server = make_server(args.host, args.port, flask_app, threaded=True)
    flask_app.extensions["server"] = http_server

    print(
        f"Client {args.client_id} listening on http://{args.host}:{args.port}",
        flush=True,
    )
    http_server.serve_forever()
