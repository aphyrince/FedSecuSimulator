import threading

import pytest

from common.model import encode_wire_weights, weights_hash
from common.training import sequential_round_schedule
from exp_manager.services.sequential_clients import SequentialClientRunner
from server.app import create_app


def test_submission_response_timeout_only_queries_server(monkeypatch, tmp_path):
    runner = SequentialClientRunner(10, tmp_path, "http://server")
    calls = []

    def post(url, update, **kwargs):
        calls.append(("POST", url, update))
        raise TimeoutError("response did not arrive")

    def get(url, **kwargs):
        calls.append(("GET", url))
        return {"status": {"0": {"status": "received"}}}

    monkeypatch.setattr("exp_manager.services.sequential_clients.post_json", post)
    monkeypatch.setattr(runner.rest, "get_json", get)
    state = runner.clients[0]
    result = runner._submit_once(state, 0, 1, {"client_id": 0})
    assert [call[0] for call in calls] == ["POST", "GET"]
    assert result["outcome"] == "response_timeout"
    assert result["server_status_after_response"]["status"] == "received"
    assert state.training_progress["status"] == "awaiting_server_confirmation"
    assert state.training_progress["error"] is None


@pytest.mark.parametrize("close_before_last_update", [False, True])
def test_sequential_round_schedules_one_aggregation_after_close_and_receipts(monkeypatch, close_before_last_update):
    app = create_app("127.0.0.1", 5000)
    app.extensions["global_model"] = {"experiment_id": "normal", "model_version": 0, "weights": {"w": [1.0]}}
    app.extensions["dataset_range"] = {"ranges": [{"file": "val", "start": 0, "end": 1}]}
    calls = []
    scheduled = threading.Event()

    def aggregate(app, round_id, payload):
        calls.append((round_id, payload))
        scheduled.set()

    monkeypatch.setattr("server.routes.aggregation._run_round_aggregation", aggregate)
    with app.test_client() as client:
        client.post("/api/clients/register", json={"clients": [
            {"client_id": client_id, "execution": "sequential"} for client_id in range(3)
        ]})
        round_url = "/api/rounds/1"
        config = {
            "experiment_id": "normal", "client_execution": "sequential",
            "participation_rate": 1.0, "nums_clients": 3,
            "assigned_samples": [{"client_id": client_id, "num_samples": 1} for client_id in range(3)],
        }
        start = client.post(f"{round_url}/start", json=config)
        assert start.status_code == 202
        assert start.get_json()["start_model_hash"] == weights_hash({"w": [1.0]})
        assert start.get_json()["selected_clients"] == [0, 1, 2]
        for client_id in range(3):
            if client_id == 2 and close_before_last_update:
                assert client.post(f"{round_url}/close", json={"experiment_id": "normal"}).status_code == 200
                assert app.extensions["round_aggregation_started"] == set()
            update = {
                "round": 1, "client_id": client_id, "num_samples": 1, "valid": True,
                "local_model": {"weights": {"w": [1.25]}}, "update": {"values": {"w": [0.25]}},
            }
            assert client.post(f"{round_url}/updates", json=update).status_code == 200
        if not close_before_last_update:
            assert app.extensions["round_aggregation_started"] == set()
        for _ in range(2):
            assert client.post(f"{round_url}/close", json={"experiment_id": "normal"}).status_code == 200
        assert scheduled.wait(2.0)
        assert len(calls) == 1
        assert app.extensions["round_aggregation_started"] == {1}
        updates = calls[0][1]["updates"]
        assert [row["client_id"] for row in updates] == [0, 1, 2]
        assert all("weights" not in row["local_model"] and "values" not in row["update"] for row in updates)
        assert sorted(app.extensions["round_update_deltas"][1]) == [0, 1, 2]


def test_half_participation_reproduces_five_distinct_clients_per_round():
    clients = list(range(10))
    plans = [{"round": index, "seed": 101, "participation_rate": 0.5} for index in (1, 2)]
    for config in plans:
        plan = sequential_round_schedule(config, clients)
        assert plan == sequential_round_schedule(config, reversed(clients))
        assert len(set(plan["selected_clients"])) == 5
        assert set(plan["unselected_clients"]) == set(clients) - set(plan["selected_clients"])


@pytest.mark.parametrize("arrival,late", [(5, []), (8, [2, 5, 8])])
def test_server_applies_saved_virtual_arrival_and_inclusive_deadline(monkeypatch, arrival, late):
    app = create_app("127.0.0.1", 5000)
    ids = list(range(10))
    app.extensions["global_model"] = {"experiment_id": "normal", "model_version": 0, "weights": {"w": [1.0]}}
    app.extensions["dataset_range"] = {"ranges": [{"file": "val", "start": 0, "end": 1}]}
    aggregated = []
    finished = threading.Event()

    def aggregate(app, round_id, payload):
        aggregated.extend(row["client_id"] for row in payload["updates"])
        finished.set()

    monkeypatch.setattr("server.routes.aggregation._run_round_aggregation", aggregate)
    config = {
        "round": 1, "experiment_id": "normal", "client_execution": "sequential",
        "participation_rate": 1.0, "nums_clients": 10, "delay_clients": [2, 5, 8],
        "virtual_timing": {"normal_arrival_seconds": 1, "delayed_arrival_seconds": arrival, "deadline_seconds": 5},
        "assigned_samples": [{"client_id": client_id, "num_samples": 1} for client_id in ids],
    }
    plan = sequential_round_schedule(config, ids)
    assert plan["virtual_late_clients"] == late
    with app.test_client() as client:
        client.post("/api/clients/register", json={"clients": [
            {"client_id": client_id, "execution": "sequential"} for client_id in ids
        ]})
        url = "/api/rounds/1"
        invalid = {**config, "sequential_schedule": {**plan, "virtual_late_clients": [-1]}}
        assert client.post(f"{url}/start", json=invalid).status_code == 400
        started = client.post(f"{url}/start", json={**config, "sequential_schedule": plan})
        assert started.status_code == 202
        assert started.get_json()["expected_clients"] == ids
        for client_id in ids:
            update = {
                "round": 1, "client_id": client_id, "num_samples": 1, "valid": True,
                "local_model": {"weights": encode_wire_weights({"w": [1.25]})},
                "update": {"values": encode_wire_weights({"w": [0.25]})},
                "tensor_encoding": "base64_float32", "virtual_arrival_seconds": 0,
            }
            response = client.post(f"{url}/updates", json=update)
            assert response.status_code == 200
            if client_id in late:
                assert response.get_json()["reason"] == "virtual_deadline_exceeded"
            else:
                assert response.get_json()["status"] == "received"
        submissions = client.get(f"{url}/updates/status").get_json()
        assert submissions["submitted_clients"] == ids and not submissions["missing_clients"]
        assert not finished.is_set()
        assert client.post(f"{url}/close", json={"experiment_id": "normal"}).status_code == 200
        assert finished.wait(2.0)
        assert aggregated == sorted(set(ids) - set(late))
