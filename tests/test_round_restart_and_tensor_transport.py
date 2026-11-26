from io import BytesIO
from unittest.mock import Mock
from urllib.error import HTTPError

import pytest
import torch

from clients.app import create_app as create_client_app
from clients.routes.experiment import _train_and_submit
from clients.services.server_client import post_json
from common.model import encode_wire_weights, weights_hash
from common.update_transport import UPDATE_CONTENT_TYPE, decode_update, encode_update
from exp_manager.services.rest_client import RestClient
from fedsec.models.resnet18 import RESNET18_FORMAT
from server.app import create_app as create_server_app
from server.routes.aggregation import _finish_round


def test_new_round_setup_waits_for_the_previous_async_round(tmp_path):
    app = create_server_app("127.0.0.1", 0)
    app.config["RESULTS_DIR"] = tmp_path
    active = {"experiment_id": "previous", "weights": {"w": [1.0]}}
    app.extensions.update(active_round_id=1, global_model=active)
    client = app.test_client()

    for endpoint, payload in (
        ("/api/clients/register", {"clients": [{"client_id": 0}]}),
        ("/api/round/config", {"experiment_id": "new", "round": 1}),
    ):
        response = client.post(endpoint, json=payload)
        assert response.status_code == 409
        assert response.json["error"] == "Round 1 is still active"
    assert client.get("/api/status").json["round_active"]
    assert app.extensions["global_model"] is active
    assert not (tmp_path / "new").exists()

    _finish_round(app, 1)
    assert not client.get("/api/status").json["round_active"]
    assert client.post("/api/clients/register", json={"clients": [{"client_id": 0}]}).status_code == 200
    response = client.post("/api/round/config", json={
        "experiment_id": "new", "round": 1, "model_config": {"channels": [2, 4], "hidden_size": 4},
    })
    assert response.status_code == 200
    assert (tmp_path / "new" / "checkpoints" / "global_0000.pt").is_file()


def test_manager_preserves_conflict_response_body(monkeypatch):
    error = HTTPError("http://server/api/clients/register", 409, "CONFLICT", {},
                      BytesIO(b'{"error":"Round 1 is still active"}'))
    monkeypatch.setattr("exp_manager.services.rest_client.urlopen", Mock(side_effect=error))
    with pytest.raises(HTTPError, match="409.*Round 1 is still active"):
        RestClient().post_json(error.url, {})


def test_client_update_failure_logs_request_size_and_rejection_reason(monkeypatch, caplog):
    url = "http://server/api/rounds/1/updates"
    error = HTTPError(url, 400, "BAD REQUEST", {}, BytesIO(b'{"error":"Submitted sample count does not match assigned data"}'))

    def reject(request, *, timeout):
        assert request.get_header("X-fedsec-client-id") == "0"
        raise error

    monkeypatch.setattr("clients.services.server_client.urlopen", reject)
    with caplog.at_level("INFO", logger="fedsec.runtime"):
        with pytest.raises(HTTPError, match="Submitted sample count does not match assigned data"):
            post_json(url, {"experiment_id": "example", "round": 1, "client_id": 0})

    assert "client sending update experiment=example round=1 client=0 bytes=" in caplog.text
    assert "client update rejected round=1 client=0 error=HTTP Error 400" in caplog.text


def test_normal_resnet_update_uses_tensor_encoding_and_keeps_summary(monkeypatch):
    app = create_server_app("127.0.0.1", 0)
    model = {
        "architecture": RESNET18_FORMAT, "model_config": {"architecture": RESNET18_FORMAT},
        "experiment_id": "resnet", "weights": {"w": [1.0]}, "model_version": 0,
    }
    app.extensions.update(
        global_model=model, selected_clients={1: [0]}, round_submission_clients={1: {0}},
        expected_sample_counts={1: {0: 16}},
    )
    local = {**model, "weights": {"w": torch.tensor([1.25])}, "model_version": 1}
    delta = {"w": torch.tensor([0.25])}
    training = {
        "round": 1, "num_samples": 16, "base_model_version": 0,
        "base_model_hash": weights_hash(model["weights"]), "dataset_fingerprint": "test-data",
        "local_model": local, "model_delta": delta,
        "extracted_update": {"values": delta, "update_norm": 0.25},
    }
    client_app = create_client_app(0, 5101)
    client_app.config["SERVER_URL"] = "http://server"
    state = client_app.extensions["client_state"]
    state.global_model = model
    client = app.test_client()

    def train(state, client_id, payload, *, tensor_payload):
        assert tensor_payload is True
        state.latest_local_training = training

    def open_update(request, *, timeout):
        assert request.get_header("Content-type") == UPDATE_CONTENT_TYPE
        damaged = bytearray(request.data)
        damaged[-1] ^= 1
        invalid = client.post("/api/rounds/1/updates", data=bytes(damaged), content_type=UPDATE_CONTENT_TYPE)
        assert invalid.status_code == 400
        assert invalid.json["error"] == "Tensor update checksum mismatch"
        assert not app.extensions.get("round_updates", {}).get(1)
        response = client.post("/api/rounds/1/updates", data=request.data, content_type=UPDATE_CONTENT_TYPE)
        assert response.status_code == 200, response.json
        response = BytesIO(response.data)
        response.status = 200
        return response

    def forbid_deepcopy(_):
        pytest.fail("ResNet submission copied parameters recursively")

    with monkeypatch.context() as patch:
        patch.setattr("clients.routes.experiment.run_local_training", train)
        patch.setattr("clients.services.server_client.urlopen", open_update)
        patch.setattr("clients.services.training_service.deepcopy", forbid_deepcopy)
        state.training_lock.acquire()
        _train_and_submit(client_app, {**model, "round": 1, "server_url": "http://server"})

    assert state.submitted_updates[1]["status"] == "received"
    assert app.extensions["round_update_deltas"][1][0]["w"].tolist() == [0.25]
    saved = app.extensions["round_updates"][1][0]
    assert saved["tensor_encoding"] == "raw_float32"
    assert "weights" not in saved["local_model"] and "values" not in saved["update"]
    assert client.get("/api/rounds/1/updates/status").status_code == 200
    assert state.latest_local_training["local_model"]["weights"] is local["weights"]
    readback = client_app.test_client().get("/api/training/status").json["training"]
    assert readback["local_model"]["weights"] == {"w": [1.25]}
    assert readback["extracted_update"]["values"] == {"w": [0.25]}


def test_binary_update_roundtrips_tensor_names_shapes_and_scalar_values():
    update = {
        "client_id": 1,
        "local_model": {"weights": {"layer.z": [[1.0, 2.0]], "scalar": 3.0}},
        "update": {"values": {"layer.z": [[0.5, 0.25]], "scalar": -1.0}},
    }
    decoded = decode_update(encode_update(update))
    assert decoded["client_id"] == 1
    for section, field in (("local_model", "weights"), ("update", "values")):
        assert {name: values.tolist() for name, values in decoded[section][field].items()} == update[section][field]


def test_client_accepts_encoded_global_model_and_retains_legacy_readback(monkeypatch):
    monkeypatch.setattr("clients.routes.experiment.threading.Thread", Mock())
    app = create_client_app(0, 5101)
    app.config["SERVER_URL"] = "http://server"
    client = app.test_client()
    weights = {"w": [1.0, 2.0]}
    response = client.post("/api/model/global", json={
        "architecture": RESNET18_FORMAT, "weights": encode_wire_weights(weights),
        "tensor_encoding": "base64_float32", "round": 1,
    })
    assert response.status_code == 202
    assert client.get("/api/model/global").json["weights"] == weights
