"""Disposable loopback benchmark for the current client update request flow.

Run with:
    .venv\\Scripts\\python.exe diagnostics\\benchmark_update_submission.py

It exercises the production Flask routes with synthetic, deterministic model
updates. It does not train a model or change production behavior.
"""

import argparse
import json
from pathlib import Path
import sys
import threading
import time
from urllib.request import Request, urlopen

from flask import Flask, g, request
from werkzeug.serving import WSGIRequestHandler, make_server

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from clients.routes.experiment import experiment_bp
from common.model import (
    MODEL_FORMAT,
    SUPPORTED_ARCHITECTURES,
    create_model,
    decode_weights,
    encode_weights,
    weights_hash,
    weights_norm,
)
from server.routes.aggregation import aggregation_bp


ROUND_ID = 1
SAMPLE_COUNT = 500


class QuietRequestHandler(WSGIRequestHandler):
    def log_request(self, code="-", size="-"):
        pass


class DiagnosticClientState:
    def __init__(self, client_id: int, global_model: dict, delta: dict) -> None:
        local_weights = {
            name: tensor + delta[name]
            for name, tensor in decode_weights(global_model["weights"]).items()
        }
        local_model = {
            **global_model,
            "weights": encode_weights(local_weights),
        }
        self.global_model = global_model
        self.expired_rounds = set()
        self.latest_local_training = {
            "round": ROUND_ID,
            "base_model_version": global_model["model_version"],
            "base_model_hash": weights_hash(global_model["weights"]),
            "local_model": local_model,
            "dataset_fingerprint": f"diagnostic-client-{client_id}",
            "num_samples": SAMPLE_COUNT,
            "training_accuracy": None,
            "training_started_at": "2026-01-01T00:00:00+00:00",
            "training_completed_at": "2026-01-01T00:00:01+00:00",
            "training_duration_seconds": 1.0,
            "extracted_update": {
                "values": encode_weights(delta),
                "update_norm": weights_norm(delta),
            },
        }
        self.submitted_updates = {}

    def save_submission(self, round_id: int, submission: dict) -> None:
        self.submitted_updates[round_id] = submission


def post_json(url: str, payload: dict) -> tuple[dict, int]:
    request_body = json.dumps(payload).encode("utf-8")
    http_request = Request(
        url,
        data=request_body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(http_request, timeout=120.0) as response:
        response_body = response.read()
        return json.loads(response_body.decode("utf-8")), len(response_body)


def start_server(app: Flask):
    server = make_server(
        "127.0.0.1",
        0,
        app,
        threaded=True,
        request_handler=QuietRequestHandler,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread, f"http://127.0.0.1:{server.server_port}"


def run_benchmark(
    client_count: int,
    architecture: str,
    include_redundant_execute: bool,
) -> None:
    base_model = create_model(seed=42, config={"architecture": architecture})
    base_model["experiment_id"] = "disposable-update-diagnostic"
    base_weights = decode_weights(base_model["weights"])
    tensor_count = sum(tensor.numel() for tensor in base_weights.values())
    server_app = Flask("diagnostic_update_server")
    server_timings = []

    @server_app.before_request
    def time_update_validation():
        if request.path == f"/api/rounds/{ROUND_ID}/updates":
            g.update_started_at = time.perf_counter()

    @server_app.after_request
    def record_update_validation(response):
        if request.path == f"/api/rounds/{ROUND_ID}/updates":
            server_timings.append(
                {
                    "seconds": time.perf_counter() - g.update_started_at,
                    "bytes": request.content_length or 0,
                }
            )
        return response

    server_app.register_blueprint(aggregation_bp)
    server_app.extensions.update(
        global_model=base_model,
        selected_clients={ROUND_ID: list(range(client_count))},
        expected_sample_counts={
            ROUND_ID: {client_id: SAMPLE_COUNT for client_id in range(client_count)}
        },
        completed_rounds=set(),
        update_deadlines={},
        update_submission_status={ROUND_ID: {}},
        update_submission_attempts={},
        round_updates={},
    )

    server_handle, server_thread, server_url = start_server(server_app)
    client_handles = []
    results = []
    try:
        for client_id in range(client_count):
            generator = torch.Generator(device="cpu").manual_seed(client_id + 42)
            delta = {
                name: torch.randn(
                    tensor.shape,
                    generator=generator,
                    dtype=tensor.dtype,
                )
                * 1e-5
                for name, tensor in base_weights.items()
            }
            client_app = Flask(f"diagnostic_client_{client_id}")
            client_app.config.update(
                CLIENT_ID=client_id,
                SERVER_URL=server_url,
            )
            client_app.extensions["client_state"] = DiagnosticClientState(
                client_id,
                base_model,
                delta,
            )
            client_app.register_blueprint(experiment_bp)
            client_handle = start_server(client_app)
            client_handles.append(client_handle)
            _, _, client_url = client_handle

            request_payload = {
                "round": ROUND_ID,
                "malicious": False,
                "attack_enabled": False,
                "server_url": server_url,
            }
            execute_response_bytes = 0
            execute_seconds = 0.0
            if include_redundant_execute:
                started_at = time.perf_counter()
                _, execute_response_bytes = post_json(
                    f"{client_url}/api/round/execute",
                    request_payload,
                )
                execute_seconds = time.perf_counter() - started_at

            started_at = time.perf_counter()
            post_json(f"{client_url}/api/round/submit", request_payload)
            submit_seconds = time.perf_counter() - started_at
            results.append(
                {
                    "client_id": client_id,
                    "execute_seconds": execute_seconds,
                    "execute_response_bytes": execute_response_bytes,
                    "submit_seconds": submit_seconds,
                }
            )

        if len(server_app.extensions["round_updates"][ROUND_ID]) != client_count:
            raise RuntimeError("Not all diagnostic updates were accepted by the server")
    finally:
        for handle in reversed(client_handles):
            server, thread, _ = handle
            server.shutdown()
            thread.join(timeout=3.0)
            server.server_close()
        server_handle.shutdown()
        server_thread.join(timeout=3.0)
        server_handle.server_close()

    print(f"architecture: {architecture}")
    print(f"parameters: {tensor_count:,}")
    print(f"clients: {client_count} (processed sequentially)")
    print(
        "request flow: "
        + ("execute + submit" if include_redundant_execute else "submit only")
    )
    print("client | execute response | execute ms | submit upload | submit ms | server validate ms")
    for item, server_timing in zip(results, server_timings):
        print(
            f"{item['client_id']:>6} | "
            f"{item['execute_response_bytes'] / 1024 / 1024:>7.2f} MiB | "
            f"{item['execute_seconds'] * 1000:>9.1f} | "
            f"{server_timing['bytes'] / 1024 / 1024:>7.2f} MiB | "
            f"{item['submit_seconds'] * 1000:>8.1f} | "
            f"{server_timing['seconds'] * 1000:>18.1f}"
        )
    total_execute = sum(item["execute_seconds"] for item in results)
    total_submit = sum(item["submit_seconds"] for item in results)
    duplicate_transfer = sum(item["execute_response_bytes"] for item in results)
    actual_upload = sum(item["bytes"] for item in server_timings)
    default_json_upload = sum(
        len(
            json.dumps(server_app.extensions["round_updates"][ROUND_ID][client_id])
            .encode("utf-8")
        )
        for client_id in range(client_count)
    )
    print(f"sequential execute time: {total_execute:.3f} s")
    print(f"sequential submit time: {total_submit:.3f} s")
    print(f"unused execute responses: {duplicate_transfer / 1024 / 1024:.2f} MiB")
    print(f"validated update uploads: {actual_upload / 1024 / 1024:.2f} MiB")
    print(
        "default JSON size for same updates: "
        f"{default_json_upload / 1024 / 1024:.2f} MiB "
        f"(compact saved {(default_json_upload - actual_upload) / 1024 / 1024:.2f} MiB)"
    )
    print("result: all synthetic submissions passed the production server validation route")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure the current update execute/submit flow over loopback."
    )
    parser.add_argument("--clients", type=int, default=5)
    parser.add_argument(
        "--architecture",
        choices=SUPPORTED_ARCHITECTURES,
        default=MODEL_FORMAT,
    )
    parser.add_argument(
        "--include-redundant-execute",
        action="store_true",
        help="Also make the now-unneeded execute request to reproduce the old flow.",
    )
    args = parser.parse_args()
    if args.clients < 1:
        parser.error("--clients must be at least 1")
    run_benchmark(
        args.clients,
        args.architecture,
        args.include_redundant_execute,
    )


if __name__ == "__main__":
    main()
