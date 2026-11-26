"""Run four E32 normal FedAvg conditions through the individual sequential runner."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import socket
from threading import Thread
import time
from uuid import uuid4

import torch
from werkzeug.serving import make_server

from exp_manager.app import create_app
from exp_manager.services.experiment_runner import CONTINUOUS_ENVIRONMENTS
from exp_manager.utils.project_root_path import PROJECT_ROOT_PATH


TRAIN_SAMPLES_PER_CLIENT = 16
CASES = (
    ("full", "baseline", 10),
    ("half", "partial", 5),
    ("delay", "delay", 7),
    ("label-bias", "label-bias", 10),
)


def _wait_for_completion(manager, experiment_id: str, timeout: float) -> dict:
    deadline = time.monotonic() + timeout
    while True:
        status = manager.status()
        if status["status"] == "completed":
            return status
        if status["status"] != "running":
            raise RuntimeError(f"{experiment_id} failed: {status['message']}")
        if time.monotonic() >= deadline:
            raise TimeoutError(f"{experiment_id} did not complete within {timeout:g} seconds")
        time.sleep(0.5)


def run_comparison(manager, data_manager, device: str) -> list[dict]:
    preset = json.loads((PROJECT_ROOT_PATH / "configs/e32_selected_fedavg.json").read_text(encoding="utf-8"))
    common = {
        **preset, "device": device, "num_rounds": 1,
        "max_train_samples": TRAIN_SAMPLES_PER_CLIENT, "max_test_samples": 64,
    }
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    batch_id = f"e32-resnet18-seq-{stamp}-{uuid4().hex[:8]}"
    data_manager.distribute(
        seed=common["seed"], distribution=common["data_distribution"],
        data_protocol=common["data_protocol"],
    )

    shared_inputs = None
    iid_assignments = None
    cases = []
    for condition, environment_key, expected_count in CASES:
        environment = CONTINUOUS_ENVIRONMENTS[environment_key]
        participation_rate = environment["participation_rate"]
        delay_clients = environment["delay_clients"]
        distribution = environment["data_distribution"]
        experiment_id = f"{batch_id}-{condition}"
        config = manager.update_config({
            **common, "experiment_id": experiment_id, "participation_rate": participation_rate,
            "delay_clients": delay_clients, "data_distribution": distribution,
        })
        print(f"Starting {experiment_id} ({participation_rate:.0%} participation)", flush=True)
        manager.start()
        status = _wait_for_completion(manager, experiment_id, config["round_timeout"])
        result_path = Path(status["result_path"])
        result = json.loads(result_path.read_text(encoding="utf-8"))
        round_result, = result["round_results"]
        if not round_result["model_updated"] or round_result["accepted_count"] != expected_count:
            raise RuntimeError(
                f"{experiment_id} did not aggregate the expected {expected_count} updates "
                f"({round_result['accepted_count']} accepted)"
            )
        if (
            round_result["trained_count"] != expected_count + len(delay_clients)
            or round_result["excluded_clients"] != delay_clients
            or any(
                row["reason"] != "virtual_deadline_exceeded"
                for row in round_result["client_activity"] if row["client_id"] in delay_clients
            )
        ):
            raise RuntimeError(f"{experiment_id} recorded unexpected training or virtual-deadline exclusions")

        directory = result_path.parent
        data_split = json.loads((directory / "experiment_info.json").read_text(encoding="utf-8"))["data_split"]
        image_set = data_split["training_image_set"]
        if image_set["sample_count"] != 10 * TRAIN_SAMPLES_PER_CLIENT:
            raise RuntimeError(f"{experiment_id} used an unexpected training pool size")
        validation = round_result["evaluation"]
        final_test = result["final_test"]
        inputs = (
            result["config"]["seed"], round_result["start_model_hash"], image_set["sha256"],
            validation["dataset_fingerprint"], final_test["dataset_fingerprint"],
        )
        if shared_inputs is not None and inputs != shared_inputs:
            raise RuntimeError(f"{experiment_id} started with a different model, seed or data split")
        shared_inputs = inputs
        assignments = data_split["client_ranges"]
        if iid_assignments is None:
            iid_assignments = assignments
        if (assignments == iid_assignments) != (distribution == "iid"):
            raise RuntimeError(f"{experiment_id} recorded an unexpected training assignment")
        cases.append({
            "experiment_id": experiment_id,
            "participation_rate": participation_rate,
            "data_distribution": distribution,
            "trained_count": round_result["trained_count"],
            "accepted_count": round_result["accepted_count"],
            "accepted_samples": round_result["accepted_samples"],
            "excluded_clients": round_result["excluded_clients"],
            "seed": result["config"]["seed"],
            "start_model_hash": round_result["start_model_hash"],
            "split_id": data_split["split_id"],
            "training_image_set_sha256": image_set["sha256"],
            "validation_samples": validation["num_test_samples"],
            "final_test_samples": final_test["num_test_samples"],
            "schedule_path": str(directory / "round_schedule.json"),
            "client_execution_path": round_result["client_execution_log"],
            "validation_path": str(directory / "rounds" / "round_001.json"),
            "final_test_path": str(directory / "final_test.json"),
        })
        print(f"Completed {experiment_id} with {expected_count} accepted updates", flush=True)
    return cases


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args(argv)
    cuda_available = torch.cuda.is_available()
    device = args.device
    if device == "auto":
        device = "cuda" if cuda_available else "cpu"
    if device == "cuda" and not cuda_available:
        parser.error("CUDA is unavailable")
    if device == "cuda":
        print(f"CUDA {torch.version.cuda} on {torch.cuda.get_device_name(0)}", flush=True)

    host = "127.0.0.1"
    with socket.socket() as port_probe:
        port_probe.bind((host, 0))
        server_port = port_probe.getsockname()[1]
        app = create_app(
            host, 0, 10, server_port, 5101, sequential_clients=True,
            e32_train_samples_per_client=TRAIN_SAMPLES_PER_CLIENT,
        )
        http_server = make_server(host, 0, app, threaded=True)
    app.extensions["server"] = http_server
    processes = app.extensions["process_manager"]
    processes.manager_port = http_server.server_port
    manager = app.extensions["experiment_manager"]
    server_thread = Thread(target=http_server.serve_forever, daemon=True)
    server_thread.start()
    try:
        processes.start()
        cases = run_comparison(manager, app.extensions["data_manager"], device)
        print(json.dumps({"device": device, "cases": cases}, indent=2), flush=True)
    finally:
        manager.stop()
        processes.stop()
        http_server.shutdown()
        server_thread.join()
        http_server.server_close()


if __name__ == "__main__":
    main()
