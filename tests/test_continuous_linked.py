import json
import tempfile
from pathlib import Path
from unittest.mock import patch

from common.model import create_model
from exp_manager.app import create_app
from exp_manager.routes.experiment import experiment_bp
from exp_manager.services.experiment_runner import (
    CONTINUOUS_ENVIRONMENTS,
    CONTINUOUS_SECURITY_SCENARIOS,
    ExperimentManager,
)
from flask import Flask


class RunningProcess:
    def poll(self):
        return None


class LinkedProcessManager:
    server_process = RunningProcess()

    def sync_client_registry(self):
        return None

    def client_registry(self):
        return [
            {"client_id": client_id, "url": f"http://127.0.0.1:{5101 + client_id}"}
            for client_id in range(10)
        ]

    def manager_url(self):
        return "http://127.0.0.1:8000"

    def server_url(self):
        return "http://127.0.0.1:5000"


def all_cases():
    return [
        {
            "environment_template": environment,
            "security_scenario": security,
        }
        for environment in CONTINUOUS_ENVIRONMENTS
        for security in CONTINUOUS_SECURITY_SCENARIOS
    ]


def linked_payload():
    return {
        "image": [0.25] * 3072,
        "label": 4,
        "dlg_iterations": 3,
        "dlg_client_id": 1,
        "noise_small": 0.01,
        "noise_large": 0.1,
        "initialization_seed": 71,
    }


def test_linked_resnet_is_rejected_before_creating_a_batch(tmp_path):
    manager = ExperimentManager(LinkedProcessManager(), tmp_path, client_count=10)
    manager.update_config({"model_config": {"architecture": "resnet18_cifar10_v1"}})
    app = Flask(__name__)
    app.extensions["experiment_manager"] = manager
    app.register_blueprint(experiment_bp)

    response = app.test_client().post("/api/experiment/continuous/linked/start", json={
        "cases": all_cases(), "dlg": linked_payload(),
    })

    assert response.status_code == 409
    assert "small_cnn_v1" in response.json["error"]
    assert manager.status()["status"] == "idle"
    assert not list(tmp_path.iterdir())


def test_selected_smallcnn_configuration_reaches_integrated_fl_rounds(tmp_path):
    manager = ExperimentManager(LinkedProcessManager(), tmp_path, client_count=10)
    app = Flask(__name__)
    app.extensions["experiment_manager"] = manager
    app.register_blueprint(experiment_bp)
    client = app.test_client()
    applied = client.post("/api/experiment/config/selected-fedavg")
    assert applied.status_code == 200
    training = {
        "model_config": {"architecture": "small_cnn_v1", "channels": [32, 64], "hidden_size": 128, "dropout": 0.0},
        "learning_rate": 0.02, "optimizer": "sgd", "momentum": 0.9, "weight_decay": 0.0,
        "augmentation": "flip_crop", "batch_size": 32, "local_epochs": 1, "device": "cuda",
    }
    assert applied.json["data_protocol"] == "e32"
    assert applied.json["use_pretrained_global_model"] is False
    with patch("exp_manager.services.experiment_runner.threading.Thread") as worker:
        response = client.post("/api/experiment/continuous/linked/start",
                               json={"cases": all_cases(), "dlg": linked_payload()})
    assert response.status_code == 200
    cases, manifest = worker.call_args.kwargs["args"][:2]
    assert len(cases) == 24
    for case, saved in zip(cases, manifest["cases"]):
        assert case["seed"] == 42 and case["num_rounds"] == 80
        for config in (manager._round_config(80, case), saved["configuration"]):
            assert {key: config[key] for key in training} == training


def test_linked_setup_failure_identifies_the_case_that_failed(tmp_path):
    manager = ExperimentManager(LinkedProcessManager(), tmp_path, client_count=10)
    with patch("exp_manager.services.experiment_runner.threading.Thread") as worker:
        manager.start_continuous_linked_experiments(all_cases(), linked_payload())
    args = worker.call_args.kwargs["args"]
    manifest_path = args[2]
    with (
        patch.object(manager, "_start_round", side_effect=lambda *_: manager.state.update(status="completed")),
        patch.object(manager, "_write_continuous_experiment_info", side_effect=[None, OSError("metadata write failed")]),
    ):
        manager._run_continuous_experiments(*args)

    saved = json.loads(manifest_path.read_text())
    batch = manager.status()["continuous_experiment"]
    assert saved["status"] == batch["status"] == "failed"
    assert saved["current_case"] == batch["current_case"] == 2
    assert batch["current_case_name"] == saved["cases"][1]["name"]
    assert [item["status"] for item in saved["cases"][:3]] == ["completed", "failed", "pending"]
    assert saved["cases"][1]["message"] == "metadata write failed"


def test_final_checkpoint_export_error_ends_the_case_and_preserves_round_log(tmp_path):
    manager = ExperimentManager(LinkedProcessManager(), tmp_path, client_count=10)
    manager.update_config({"experiment_id": "output-failure", "num_rounds": 1})
    manager.state["status"] = "running"
    with patch.object(manager, "_save_final_global_model_for_dlg", side_effect=OSError("disk full")):
        path = manager.record_round_result(
            {"experiment_id": "output-failure", "round": 1, "accuracy": 0.5, "loss": 1.0},
            advance=True,
        )
    assert json.loads(path.read_text())["accuracy"] == 0.5
    assert manager.status()["status"] == "failed"
    assert "disk full" in manager.status()["message"]


def test_manifest_write_failure_still_ends_batch_in_status_api(tmp_path):
    manager = ExperimentManager(LinkedProcessManager(), tmp_path, client_count=10)
    with patch("exp_manager.services.experiment_runner.threading.Thread") as worker:
        manager.start_continuous_linked_experiments(all_cases(), linked_payload())
    with patch.object(manager, "_write_continuous_manifest", side_effect=OSError("manifest unavailable")):
        manager._run_continuous_experiments(*worker.call_args.kwargs["args"])

    status = manager.status()
    assert status["status"] == status["continuous_experiment"]["status"] == "failed"
    assert status["continuous_experiment"]["cases"][0]["message"] == "manifest unavailable"


def test_linked_start_creates_24_fl_and_12_normal_checkpoint_dlg_cases():
    with tempfile.TemporaryDirectory() as temporary:
        manager = ExperimentManager(
            LinkedProcessManager(), Path(temporary), client_count=10
        )
        with patch("exp_manager.services.experiment_runner.threading.Thread") as thread:
            status = manager.start_continuous_linked_experiments(
                all_cases(), linked_payload()
            )

        batch = status["continuous_experiment"]
        manifest = json.loads(
            Path(batch["manifest_path"]).read_text(encoding="utf-8")
        )
        assert batch["experiment_type"] == "federated_dlg"
        assert batch["total_cases"] == 36
        assert manifest["experiment_type"] == "federated_dlg"
        assert len(manifest["cases"]) == 36
        assert sum(case["experiment_type"] == "federated" for case in manifest["cases"]) == 24
        dlg_cases = [
            case for case in manifest["cases"] if case["experiment_type"] == "dlg"
        ]
        integrated_dir = Path(batch["manifest_path"]).parent
        assert len(dlg_cases) == 12
        assert len({case["experiment_id"] for case in manifest["cases"]}) == 36
        assert integrated_dir.name == batch["batch_id"]
        assert (integrated_dir / "cases").is_dir()
        assert {case["environment_template"] for case in dlg_cases} == set(
            CONTINUOUS_ENVIRONMENTS
        )
        for case in manifest["cases"][:24]:
            assert (integrated_dir / case["result_directory"] / "experiment_info.json").is_file()
            assert not (Path(temporary) / case["experiment_id"]).exists()
        assert {case["parent_experiment_id"] for case in dlg_cases} == {
            case["experiment_id"]
            for case in manifest["cases"]
            if case["experiment_type"] == "federated"
            and case["security_scenario"] == "normal"
        }
        assert {case["security_scenario"] for case in dlg_cases} == {
            "attack",
            "attack_defense_noise_small",
            "attack_defense_noise_large",
        }
        assert all(case["status"] == "pending" for case in dlg_cases)
        saved_inputs = json.loads(
            (Path(batch["manifest_path"]).parent / "dlg_inputs.json").read_text(
                encoding="utf-8"
            )
        )
        assert saved_inputs["image"] == linked_payload()["image"]
        assert saved_inputs["label"] == linked_payload()["label"]
        assert saved_inputs["initialization_seed"] == 71
        assert thread.call_args.kwargs["args"][-1]["dlg_client_id"] == 1


def test_linked_runner_runs_dlg_triplets_only_for_normal_fl_cases():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        manager = ExperimentManager(
            LinkedProcessManager(), root, client_count=10
        )
        with patch("exp_manager.services.experiment_runner.threading.Thread") as thread:
            manager.start_continuous_linked_experiments(
                all_cases(), linked_payload()
            )
        cases, manifest, manifest_path, base_config, dlg_options = (
            thread.call_args.kwargs["args"]
        )
        order = []
        observed_configs = []

        def finish_case(config, index, active_manifest, _manifest_path):
            order.append(config["experiment_type"])
            observed_configs.append(config)
            active_manifest["cases"][index - 1]["status"] = "completed"
            manager.state["continuous_experiment"]["cases"][index - 1][
                "status"
            ] = "completed"
            manager.state.update(status="completed", message="done")
            if config["experiment_type"] == "federated":
                model = create_model(seed=index, config={"channels": [1, 1], "hidden_size": 2})
                model["experiment_id"] = config["experiment_id"]
                output = (
                    root
                    / config["integrated_experiment_id"]
                    / "cases"
                    / config["experiment_id"]
                    / "global_model_for_dlg.json"
                )
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_text(json.dumps(model))
            return True

        with patch.object(manager, "_run_continuous_case", side_effect=finish_case):
            manager._run_continuous_experiments(
                cases, manifest, manifest_path, base_config, dlg_options
            )

        assert order[:24] == ["federated"] * 24
        assert order[24:] == ["dlg"] * 12
        saved = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert saved["status"] == "completed"
        assert len(saved["cases"]) == 36
        assert all(case["status"] == "completed" for case in saved["cases"])
        for start in range(24, 36, 3):
            trio = saved["cases"][start : start + 3]
            assert len({case["parent_experiment_id"] for case in trio}) == 1
            assert trio[0]["parent_experiment_id"] in {
                case["experiment_id"]
                for case in saved["cases"][:24]
                if case["security_scenario"] == "normal"
            }
            assert [case["security_scenario"] for case in trio] == [
                "attack",
                "attack_defense_noise_small",
                "attack_defense_noise_large",
            ]
        dlg_configs = observed_configs[24:]
        assert len(dlg_configs) == 12
        for start in range(0, len(dlg_configs), 3):
            trio = dlg_configs[start : start + 3]
            assert {case["dlg_initialization_seed"] for case in trio} == {71}
            assert {tuple(case["dlg_image"]) for case in trio} == {
                tuple(linked_payload()["image"])
            }
            assert {case["dlg_label"] for case in trio} == {4}
            assert len(
                {
                    case["pretrained_global_model"]["initial_model_hash"]
                    for case in trio
                }
            ) == 1


def test_linked_api_and_results_summary_expose_a_single_dashboard():
    with tempfile.TemporaryDirectory() as temporary:
        manager = ExperimentManager(
            LinkedProcessManager(), Path(temporary), client_count=10
        )
        app = Flask(__name__)
        app.extensions["experiment_manager"] = manager
        app.register_blueprint(experiment_bp)
        with patch("exp_manager.services.experiment_runner.threading.Thread"):
            response = app.test_client().post(
                "/api/experiment/continuous/linked/start",
                json={"cases": all_cases(), "dlg": linked_payload()},
            )
        assert response.status_code == 200
        assert response.json["continuous_experiment"]["total_cases"] == 36
        ui = create_app("127.0.0.1", 0, 10, 5000, 5101).test_client().get("/")
        assert ui.status_code == 200
        page = ui.get_data(as_text=True)
        assert 'id="continuous-linked-mode-panel"' in page
        assert 'id="continuous-linked-experiments"' in page
        assert 'id="start-continuous-linked-button"' in page
        assert 'id="continuous-linked-progress"' in page
        assert 'role="progressbar"' in page
        assert 'id="continuous-linked-case-progress-list"' in page
        assert "실험별 진행률" in page
        assert 'id="continuous-results-rows"' in page
        for setting_id in (
            "rounds",
            "seed",
            "local-epochs",
            "batch-size",
            "learning-rate",
            "device",
            "round-timeout",
            "max-train-samples",
            "max-test-samples",
            "malicious-clients",
        ):
            assert f'id="linked-shared-{setting_id}"' in page
        assert 'id="linked-shared-participation"' not in page
        assert "참여율은 라운드별 설정에서 조정" in page

        manager.state["continuous_experiment"] = None
        batch_dir = Path(temporary) / "integrated-summary"
        result_dir = batch_dir / "cases" / "summary-case"
        result_dir.mkdir(parents=True)
        (result_dir / "results.json").write_text(
            json.dumps({"round_results": [{"accuracy": 0.75, "loss": 0.4}]})
        )
        (batch_dir / "manifest.json").write_text(
            json.dumps(
                {
                    "batch_id": "batch-summary",
                    "experiment_type": "federated_dlg",
                    "status": "completed",
                    "started_at": "2099-01-01T00:00:00+00:00",
                    "cases": [
                        {
                            "experiment_id": "summary-case",
                            "experiment_type": "federated",
                            "environment_label": "기준",
                            "security_label": "정상",
                            "status": "completed",
                            "result_directory": "cases/summary-case",
                        }
                    ],
                }
            )
        )
        summary = app.test_client().get("/api/results/continuous")
        assert summary.status_code == 200
        summary_case = next(
            case
            for case in summary.json["cases"]
            if case["experiment_id"] == "summary-case"
        )
        assert summary_case["accuracy"] == 0.75


def test_integrated_results_summary_prefers_active_then_latest_batch():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        manager = ExperimentManager(
            LinkedProcessManager(), root, client_count=10
        )
        for batch_id, started_at, status in (
            ("integrated-older", "2026-11-14T08:00:00+00:00", "running"),
            ("integrated-latest", "2026-11-14T09:00:00+00:00", "completed"),
        ):
            batch_dir = root / batch_id
            case_dir = batch_dir / "cases" / f"{batch_id}-case"
            case_dir.mkdir(parents=True)
            (case_dir / "results.json").write_text(
                json.dumps({"round_results": [{"accuracy": 0.75, "loss": 0.4}]})
            )
            (batch_dir / "manifest.json").write_text(
                json.dumps(
                    {
                        "batch_id": batch_id,
                        "experiment_type": "federated_dlg",
                        "status": status,
                        "started_at": started_at,
                        "total_cases": 36,
                        "cases": [
                            {
                                "experiment_id": f"{batch_id}-case",
                                "experiment_type": "federated",
                                "environment_template": "baseline",
                                "environment_label": "기준",
                                "security_scenario": "normal",
                                "security_label": "정상",
                                "status": "completed",
                                "result_directory": f"cases/{batch_id}-case",
                            }
                        ],
                    }
                )
            )

        manager.state["continuous_experiment"] = {
            "batch_id": "integrated-older",
            "experiment_type": "federated_dlg",
            "status": "running",
        }
        summary = manager.continuous_results_summary()
        assert summary["batch"]["batch_id"] == "integrated-older"
        assert len(summary["cases"]) == 1

        manager.state["continuous_experiment"] = None
        summary = manager.continuous_results_summary()
        assert summary["batch"]["batch_id"] == "integrated-latest"
        assert len(summary["cases"]) == 1
