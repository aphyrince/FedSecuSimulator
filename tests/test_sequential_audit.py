import json
import time

import pytest
import torch
from torch.utils.data import TensorDataset

from common.model import create_model, save_model_checkpoint, weights_hash
from common.training import sequential_round_schedule
from exp_manager.services.experiment_runner import ExperimentManager
from exp_manager.services.sequential_clients import SequentialClientRunner
from server.app import create_app
from server.services.model_evaluation import evaluate_global_model


@pytest.mark.parametrize("accepted_count", [0, 1, 2])
def test_sequential_round_below_minimum_keeps_model(monkeypatch, tmp_path, accepted_count):
    app = create_app("127.0.0.1", 5000)
    app.config["RESULTS_DIR"] = tmp_path
    weights = {"w": [1.0]}
    start_hash = weights_hash(weights)
    model = {"experiment_id": "small-cohort", "model_version": 3, "weights": weights, "initial_model_hash": start_hash}
    app.extensions["global_model"] = model
    app.extensions["dataset_range"] = {"ranges": [{"file": "val", "start": 0, "end": 1}]}

    def evaluate(actual_model, dataset, config, round_id):
        assert actual_model is model
        return {"accuracy": 1.0, "loss": 0.1, "model_hash": start_hash, "model_version": 3}

    monkeypatch.setattr("server.routes.aggregation.evaluate_global_model", evaluate)
    clients = list(range(10))
    with app.test_client() as client:
        client.post("/api/clients/register", json={"clients": [
            {"client_id": client_id, "execution": "sequential"} for client_id in clients
        ]})
        url = "/api/rounds/1"
        config = {
            "experiment_id": model["experiment_id"], "client_execution": "sequential",
            "participation_rate": 0.5, "nums_clients": len(clients), "seed": 101,
            "assigned_samples": [{"client_id": client_id, "num_samples": 2} for client_id in clients],
        }
        start = client.post(f"{url}/start", json=config)
        assert start.status_code == 202
        selected = start.get_json()["selected_clients"]
        for client_id in selected[:accepted_count]:
            update = {
                "round": 1, "client_id": client_id, "num_samples": 2, "valid": True,
                "local_model": {"weights": {"w": [1.25]}, "model_version": 4},
                "update": {"values": {"w": [0.25]}},
            }
            assert client.post(f"{url}/updates", json=update).status_code == 200
        for client_id in selected[accepted_count:]:
            assert client.post(f"{url}/failures", json={
                "client_id": client_id, "reason": "real_training_failure",
            }).status_code == 200
        assert client.post(f"{url}/close", json={"experiment_id": model["experiment_id"]}).status_code == 200
        deadline = time.monotonic() + 3
        while not app.extensions["round_results"] and time.monotonic() < deadline:
            time.sleep(0.005)
        result, = app.extensions["round_results"]
        assert result["reason"] == "insufficient_accepted_updates"
        assert result["accepted_count"] == accepted_count
        assert result["accepted_samples"] == accepted_count * 2
        assert result["fedavg_samples"] is None and result["aggregated_delta"] is None
        assert not result["model_updated"]
        assert result["start_model_version"] == result["end_model_version"] == 3
        assert result["start_model_hash"] == result["end_model_hash"] == start_hash
        assert client.get("/api/model/history").get_json()["count"] == 0
        end_model = client.get("/api/model/global").get_json()
        assert end_model == model
        assert weights_hash(end_model["weights"]) == start_hash
        checkpoint = torch.load(result["global_model"]["checkpoint"], map_location="cpu", weights_only=True)
        assert checkpoint["model_hash"] == start_hash and checkpoint["model_version"] == 3
        for row in result["client_activity"]:
            if row["client_id"] not in selected[:accepted_count]:
                assert all(row[key] is None for key in ("num_samples", "base_model_version", "local_model_hash", "update_norm"))
                assert row["reason"] == ("real_training_failure" if row["client_id"] in selected else "not_selected")
                assert bool(row["excluded_at"]) == (row["client_id"] in selected)


def test_class_counts_and_fixed_client_evaluation_shards(monkeypatch):
    labels = torch.tensor([0, 1, 1])
    logits = torch.tensor([[4., 0.], [0., 4.], [4., 0.]])

    def dataset(_ranges, limit):
        count = min(limit or len(labels), len(labels))
        result = TensorDataset(torch.arange(count), labels[:count])
        result.fingerprint = f"validation-{count}"
        return result

    class FixedModel(torch.nn.Module):
        def forward(self, indices):
            return logits[indices]

    monkeypatch.setattr("server.services.model_evaluation.CifarRangeDataset", dataset)
    monkeypatch.setattr("server.services.model_evaluation.build_model", lambda _: FixedModel())
    model = {"experiment_id": "held-out", "model_version": 1, "weights": {"w": [1.0]}}
    config = {"client_evaluation_ids": [0, 1, 2, 3], "client_evaluation_max_samples": 3}
    first = evaluate_global_model(model, {"split": "validation"}, {**config, "max_test_samples": 2}, 1)
    second = evaluate_global_model(model, {"split": "validation"}, {**config, "max_test_samples": 3}, 2)
    assert (first["num_test_samples"], second["num_test_samples"]) == (2, 3)
    assert sum(row["num_samples"] for row in second["metrics"]["per_class"]) == 3
    assert sum(row["correct_predictions"] for row in second["metrics"]["per_class"]) == 2
    assert second["metrics"]["per_class"][2]["accuracy"] is None
    assert [r["dataset_fingerprint"] for r in first["client_evaluations"]] == [r["dataset_fingerprint"] for r in second["client_evaluations"]]
    assert [r["num_test_samples"] for r in second["client_evaluations"]] == [1, 1, 1, None]
    assert [r["correct_predictions"] for r in second["client_evaluations"]] == [1, 1, 0, None]
    assert second["client_evaluations"][-1]["accuracy"] is None


def test_replay_preserves_later_round_number_start_version_and_schedule(monkeypatch, tmp_path):
    class ProcessManager:
        def __init__(self):
            self.sequential_runner = SequentialClientRunner(3, tmp_path, "http://server")

        def server_url(self):
            return "http://server"

        def sync_client_registry(self):
            pass

    processes = ProcessManager()
    manager = ExperimentManager(processes, tmp_path, 3)
    config = manager.get_config()
    config.update(experiment_id="source", seed=101, num_rounds=3)
    source = tmp_path / "source"
    checkpoint = source / "checkpoints" / "round_0002_start.pt"
    model = create_model(seed=101, config=config["model_config"])
    model.update(model_version=1, experiment_id="source")
    save_model_checkpoint(model, checkpoint)
    data = {"source_dir": str(tmp_path), "ranges": [{"file": "validation", "start": 0, "end": 3}], "split": "validation"}
    split = {"source_dir": str(tmp_path), "test_range": data, "validation_range": data,
             "client_ranges": [{"client_id": i, "ranges": [{"file": "train", "start": i, "end": i + 1}]} for i in range(3)]}
    (source / "experiment_info.json").write_text(json.dumps({"configuration": config, "data_split": split}))
    original_config = {
        **manager._round_config(2, config), "round": 2, "seed": 101, "client_execution": "sequential", "final_test_round": 3,
    }
    plan = sequential_round_schedule(original_config, range(3))
    original_config["sequential_schedule"] = plan
    saved_schedule = {"seed": 101, "registered_clients": [0, 1, 2], "rounds": [plan]}
    (source / "round_schedule.json").write_text(json.dumps(saved_schedule))
    (source / "rounds").mkdir()
    (source / "rounds" / "round_002.json").write_text(json.dumps({
        "round_config": original_config, "start_checkpoint": str(checkpoint),
        "start_model_hash": weights_hash(model["weights"]), "start_model_version": 1, "selected_clients": [0, 1, 2],
    }))
    posts = []
    monkeypatch.setattr(manager.rest_client, "post_json", lambda url, payload: posts.append((url, payload)))
    monkeypatch.setattr(manager, "_begin_run", lambda **options: options)
    replay = manager.replay_round("source", 2)
    assert replay["first_round"] == 2 and replay["initial_model"]["model_version"] == 1
    assert replay["saved_round_config"]["final_test_round"] == 3
    assert replay["saved_round_config"]["sequential_schedule"] == plan
    assert posts == [("http://server/api/dataset/range", data)]
    assert manager.get_config()["num_rounds"] == 2
    output = tmp_path / manager.get_config()["experiment_id"]
    assert output != source and json.loads((output / "round_schedule.json").read_text()) == saved_schedule
