import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
from flask import Flask

from clients.app import create_app as create_client_app
from common.model import create_model
from exp_manager.app import create_app as create_manager_app
from exp_manager.routes.experiment import experiment_bp
from exp_manager.services.experiment_runner import ExperimentManager


class RunningProcess:
    def poll(self):
        return None


class DLGProcessManager:
    server_process = RunningProcess()

    def sync_client_registry(self):
        return None

    def client_registry(self):
        return [
            {
                "client_id": client_id,
                "url": f"http://127.0.0.1:{5101 + client_id}",
            }
            for client_id in range(3)
        ]

    def manager_url(self):
        return "http://127.0.0.1:8000"

    def server_url(self):
        return "http://127.0.0.1:5000"


def small_checkpoint():
    return create_model(seed=9, config={"channels": [1, 1], "hidden_size": 2})


def start_payload():
    return {
        "checkpoint_data": json.dumps(small_checkpoint()),
        "image": [0.25] * (3 * 32 * 32),
        "label": 4,
        "dlg_iterations": 5,
        "dlg_client_id": 1,
        "noise_small": 0.01,
        "noise_large": 0.1,
    }


def test_continuous_dlg_cases_share_uploaded_checkpoint_image_and_label():
    with tempfile.TemporaryDirectory() as temporary_directory:
        manager = ExperimentManager(
            DLGProcessManager(), Path(temporary_directory), client_count=3
        )
        with patch("exp_manager.services.experiment_runner.threading.Thread") as thread:
            status = manager.start_continuous_dlg_experiments(start_payload())

        assert status["continuous_experiment"]["experiment_type"] == "dlg"
        assert status["continuous_experiment"]["total_cases"] == 3
        cases, manifest, manifest_path, base_config = thread.call_args.kwargs["args"]
        assert [case["continuous_scenario"]["security_scenario"] for case in cases] == [
            "attack",
            "attack_defense_noise_small",
            "attack_defense_noise_large",
        ]
        assert [case["dlg_defense_enabled"] for case in cases] == [False, True, True]
        assert [case["dlg_noise_sigma"] for case in cases] == [0, 0.01, 0.1]
        assert {case["dlg_initialization_seed"] for case in cases} == {
            manager.config["seed"] + 1
        }
        assert all(case["experiment_type"] == "dlg" for case in cases)
        assert all(case["use_pretrained_global_model"] for case in cases)
        assert all(case["dlg_client_id"] == 1 for case in cases)
        assert all(case["dlg_label"] == 4 for case in cases)
        assert all(case["dlg_image"] == [0.25] * 3072 for case in cases)
        checkpoint_hashes = {
            case["pretrained_global_model"]["initial_model_hash"] for case in cases
        }
        assert len(checkpoint_hashes) == 1
        assert all(
            case["pretrained_global_model"]["weights"]
            == cases[0]["pretrained_global_model"]["weights"]
            for case in cases
        )
        assert base_config["experiment_type"] == "federated"
        assert manifest["experiment_type"] == "dlg"
        assert manifest["original_label"] == 4
        for case in manifest["cases"]:
            info_path = Path(temporary_directory) / case["experiment_id"] / "experiment_info.json"
            info = json.loads(info_path.read_text(encoding="utf-8"))
            assert info["experiment_type"] == "dlg"
            assert info["configuration"]["dlg_label"] == 4
            assert info["configuration"]["pretrained_global_model_hash"] in checkpoint_hashes
            assert "dlg_image" not in info["configuration"]
            assert "pretrained_global_model" not in info["configuration"]
            assert case["configuration"]["original_label"] == 4
            assert case["configuration"]["checkpoint_hash"] in checkpoint_hashes


def test_continuous_dlg_runner_uses_each_prepared_case_and_finishes_batch():
    with tempfile.TemporaryDirectory() as temporary_directory:
        manager = ExperimentManager(
            DLGProcessManager(), Path(temporary_directory), client_count=3
        )
        with patch("exp_manager.services.experiment_runner.threading.Thread") as thread:
            manager.start_continuous_dlg_experiments(start_payload())
        cases, manifest, manifest_path, base_config = thread.call_args.kwargs["args"]
        observed = []

        def finish_case(config, round_id):
            observed.append((config, round_id))
            manager.state.update(status="completed", current_round=round_id)

        with patch.object(manager, "_start_round", side_effect=finish_case):
            manager._run_continuous_experiments(
                cases, manifest, manifest_path, base_config
            )

        assert len(observed) == 3
        assert [round_id for _, round_id in observed] == [1, 1, 1]
        assert all(config["dlg_image"] == cases[0]["dlg_image"] for config, _ in observed)
        assert all(config["dlg_label"] == cases[0]["dlg_label"] for config, _ in observed)
        assert all(
            config["pretrained_global_model"]["initial_model_hash"]
            == cases[0]["pretrained_global_model"]["initial_model_hash"]
            for config, _ in observed
        )
        saved = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert saved["status"] == "completed"
        assert [case["status"] for case in saved["cases"]] == ["completed"] * 3
        assert all(
            case["result_path"].endswith("dlg_result.json")
            for case in saved["cases"]
        )
        assert manager.get_config()["experiment_type"] == "federated"


def test_continuous_dlg_result_persists_the_shared_original_label_and_image():
    with tempfile.TemporaryDirectory() as temporary_directory:
        manager = ExperimentManager(
            DLGProcessManager(), Path(temporary_directory), client_count=3
        )
        with patch("exp_manager.services.experiment_runner.threading.Thread") as thread:
            manager.start_continuous_dlg_experiments(start_payload())
        config = thread.call_args.kwargs["args"][0][0]
        manager.config = config
        manager.dlg_participant_by_round[1] = 1
        manager.store_dlg_original(
            1,
            config["experiment_id"],
            1,
            config["dlg_image"],
            config["dlg_label"],
        )
        manager.dlg_checkpoints_by_round[1] = config["pretrained_global_model"]
        result_path = manager.record_dlg_result(
            1,
            {
                "client_id": 1,
                "reconstructed_image": config["dlg_image"],
                "iterations": config["dlg_iterations"],
            },
        )

        result = json.loads(result_path.read_text(encoding="utf-8"))
        assert result["original_label"] == 4
        assert result["original_image"] == config["dlg_image"]
        assert result["experiment_id"] == config["experiment_id"]
        replay_inputs = json.loads(
            (result_path.parent / "dlg_inputs.json").read_text(encoding="utf-8")
        )
        assert replay_inputs["image"] == config["dlg_image"]
        assert replay_inputs["label"] == config["dlg_label"]
        assert (result_path.parent / "inputs" / "model.pt").is_file()
        assert replay_inputs["image_sha256"]
        manager.state["status"] = "completed"
        manager.state["continuous_experiment"]["status"] = "completed"
        with patch.object(manager, "execute_single_round", return_value={"round": 1}) as execute:
            replayed = manager.replay_dlg(config["experiment_id"])
        replay_config = manager.get_config()
        assert replayed == {"round": 1}
        assert replay_config["dlg_image"] == config["dlg_image"]
        assert replay_config["dlg_label"] == config["dlg_label"]
        assert replay_config["dlg_client_id"] == 1
        assert execute.call_args.args[1]["round_type"] == "dlg"


def test_continuous_dlg_sends_original_only_to_client():
    with tempfile.TemporaryDirectory() as temporary_directory:
        process_manager = DLGProcessManager()
        manager = ExperimentManager(
            process_manager, Path(temporary_directory), client_count=3
        )
        with patch("exp_manager.services.experiment_runner.threading.Thread") as thread:
            manager.start_continuous_dlg_experiments(start_payload())
        cases = thread.call_args.kwargs["args"][0]
        config = cases[0]
        manager.config = config
        requests = []

        class CaptureRestClient:
            def post_json(self, url, payload, **kwargs):
                requests.append((url, payload))
                return {"status": "configured" if "/round/config" in url else "submitted"}

        manager.rest_client = CaptureRestClient()
        round_config = manager._round_config(1, config)
        round_config["experiment_id"] = config["experiment_id"]
        manager.execute_dlg_round(1, round_config)

        assert "/api/round/config" in requests[0][0]
        assert "dlg_image" not in requests[0][1]
        assert "dlg_label" not in requests[0][1]
        assert requests[1][0].endswith("/api/dlg/start")
        assert requests[1][1]["dlg_image"] == config["dlg_image"]
        assert requests[1][1]["dlg_label"] == config["dlg_label"]
        assert (
            requests[1][1]["global_model"]["weights"]
            == config["pretrained_global_model"]["weights"]
        )


def test_continuous_dlg_rejects_invalid_shared_inputs():
    with tempfile.TemporaryDirectory() as temporary_directory:
        manager = ExperimentManager(
            DLGProcessManager(), Path(temporary_directory), client_count=3
        )
        invalid = start_payload()
        invalid["image"] = [0.5]
        try:
            manager.start_continuous_dlg_experiments(invalid)
        except ValueError as error:
            assert "원본 이미지" in str(error)
        else:
            raise AssertionError("Invalid image was accepted")

        invalid = start_payload()
        invalid["noise_small"] = 0.1
        invalid["noise_large"] = 0.1
        try:
            manager.start_continuous_dlg_experiments(invalid)
        except ValueError as error:
            assert "노이즈 값" in str(error)
        else:
            raise AssertionError("Non-increasing noise levels were accepted")


def test_continuous_dlg_endpoint_starts_uploaded_batch():
    with tempfile.TemporaryDirectory() as temporary_directory:
        manager = ExperimentManager(
            DLGProcessManager(), Path(temporary_directory), client_count=3
        )
        app = Flask(__name__)
        app.extensions["experiment_manager"] = manager
        app.register_blueprint(experiment_bp)
        with patch("exp_manager.services.experiment_runner.threading.Thread"):
            response = app.test_client().post(
                "/api/experiment/continuous/dlg/start", json=start_payload()
            )

        assert response.status_code == 200
        assert response.json["continuous_experiment"]["total_cases"] == 3
        assert response.json["continuous_experiment"]["experiment_type"] == "dlg"


def test_manager_page_exposes_continuous_dlg_inputs():
        app = create_manager_app("127.0.0.1", 0, 3, 5000, 5101)
        page = app.test_client().get("/").get_data(as_text=True)
        assert 'id="continuous-dlg-checkpoint"' in page
        assert 'id="continuous-dlg-original-image"' in page
        assert 'id="continuous-dlg-label"' in page
        assert 'id="start-continuous-dlg-button"' in page


def test_client_uses_submitted_image_and_label_for_real_cnn_dlg():
    app = create_client_app(client_id=1, port=5102)
    app.config["SERVER_URL"] = "http://127.0.0.1:5000"
    pixels = [index / 3071 for index in range(3072)]
    captured = {}

    def capture_gradient(model, image, label, **kwargs):
        captured["image"] = image.detach().cpu()
        captured["label"] = label
        return {"format": "small_cnn_gradient_v1", "values": [], "noise_sigma": 0}

    def capture_post(url, payload, **kwargs):
        captured.setdefault("posts", []).append((url, payload))
        return {"status": "completed"}

    payload = {
        "global_model": {
            "architecture": "small_cnn_v1",
            "model_config": {"channels": [1, 1], "hidden_size": 2},
            "weights": {},
            "experiment_id": "continuous-dlg-test",
        },
        "dlg_image": pixels,
        "dlg_label": 6,
        "round": 1,
        "experiment_manager_url": "http://127.0.0.1:8000",
        "server_url": "http://127.0.0.1:5000",
    }
    with (
        patch("security.dlg_api.make_gradient_payload", side_effect=capture_gradient),
        patch("clients.routes.experiment.post_json", side_effect=capture_post),
    ):
        response = app.test_client().post("/api/dlg/start", json=payload)

    assert response.status_code == 200
    assert captured["label"] == 6
    assert captured["image"].shape == (3, 32, 32)
    assert abs(float(captured["image"][0, 0, 0]) - (-1.0)) < 1e-6
    assert captured["posts"][0][1]["image"] == pytest.approx(pixels, abs=1e-6)
    assert captured["posts"][0][1]["label"] == 6
