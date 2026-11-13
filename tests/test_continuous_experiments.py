import json
import tempfile
import threading
from pathlib import Path
from unittest.mock import patch

import pytest

from exp_manager.app import create_app
from exp_manager.services.experiment_runner import (
    CONTINUOUS_CASE_COUNT,
    CONTINUOUS_ENVIRONMENTS,
    CONTINUOUS_SECURITY_SCENARIOS,
    ExperimentManager,
)


class RunningProcess:
    def poll(self):
        return None


class ProcessManager:
    server_process = RunningProcess()

    def sync_client_registry(self):
        return None


def scenario_cases():
    return [
        {
            "environment_template": environment,
            "security_scenario": security,
            "name": (
                f"{CONTINUOUS_ENVIRONMENTS[environment]['label']} · "
                f"{CONTINUOUS_SECURITY_SCENARIOS[security]['label']}"
            ),
        }
        for environment in CONTINUOUS_ENVIRONMENTS
        for security in CONTINUOUS_SECURITY_SCENARIOS
    ]


def test_continuous_start_creates_24_unique_experiments_and_metadata():
    with tempfile.TemporaryDirectory() as temporary_directory:
        results_dir = Path(temporary_directory)
        manager = ExperimentManager(ProcessManager(), results_dir, client_count=10)
        manager.config.update(experiment_type="federated", num_rounds=3)
        with patch("exp_manager.services.experiment_runner.threading.Thread") as thread:
            status = manager.start_continuous_experiments(scenario_cases())

        batch = status["continuous_experiment"]
        assert batch["total_cases"] == CONTINUOUS_CASE_COUNT == 24
        assert batch["status"] == "running"
        ids = [case["experiment_id"] for case in batch["cases"]]
        assert len(set(ids)) == CONTINUOUS_CASE_COUNT
        by_scenario = {
            (case["environment_template"], case["security_scenario"]): case
            for case in batch["cases"]
        }
        assert by_scenario[("baseline", "normal")]["configuration"][
            "participation_rate"
        ] == 1
        assert by_scenario[("delay", "normal")]["configuration"][
            "delay_clients"
        ] == [2, 5, 8]
        assert by_scenario[("partial", "normal")]["configuration"][
            "participation_rate"
        ] == 0.5
        assert by_scenario[("label-bias", "normal")]["configuration"][
            "data_distribution"
        ] == "non_iid"
        assert by_scenario[("baseline", "defense_norm_clipping")]["configuration"][
            "defense_enabled"
        ] is True
        assert by_scenario[("baseline", "defense_norm_clipping")]["configuration"][
            "defense"
        ] == "norm_clipping"
        assert by_scenario[("baseline", "defense_coordinate_median")]["configuration"][
            "defense"
        ] == "coordinate_median"
        assert by_scenario[("baseline", "attack_defense_norm_clipping")]["configuration"][
            "attack_enabled"
        ] is True
        assert by_scenario[("baseline", "attack_defense_norm_clipping")]["configuration"][
            "defense"
        ] == "norm_clipping"
        assert by_scenario[("baseline", "attack_defense_coordinate_median")]["configuration"][
            "attack_enabled"
        ] is True
        assert by_scenario[("baseline", "attack_defense_coordinate_median")]["configuration"][
            "defense"
        ] == "coordinate_median"
        manifest_path = Path(batch["manifest_path"])
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert manifest["batch_id"] == batch["batch_id"]
        assert len(manifest["cases"]) == CONTINUOUS_CASE_COUNT
        for case in manifest["cases"]:
            info_path = results_dir / case["experiment_id"] / "experiment_info.json"
            info = json.loads(info_path.read_text(encoding="utf-8"))
            assert info["experiment_id"] == case["experiment_id"]
            assert info["scenario"]["environment_label"] == case["environment_label"]
            assert info["scenario"]["security_label"] == case["security_label"]
            assert info["configuration"]["num_rounds"] == 3
        thread.assert_called_once()


def test_continuous_worker_runs_cases_in_order_and_persists_outcomes():
    with tempfile.TemporaryDirectory() as temporary_directory:
        results_dir = Path(temporary_directory)
        manager = ExperimentManager(ProcessManager(), results_dir, client_count=10)
        manager.config.update(experiment_type="federated", num_rounds=1)
        batch_id = "batch-test"
        cases = manager._prepare_continuous_cases(
            scenario_cases(), batch_id, manager.get_config()
        )
        manifest_path = results_dir / "continuous_runs" / batch_id / "manifest.json"
        manifest_path.parent.mkdir(parents=True)
        manifest = {
            "batch_id": batch_id,
            "status": "running",
            "total_cases": CONTINUOUS_CASE_COUNT,
            "current_case": 0,
            "cases": [
                manager._continuous_case_manifest(index, config, "pending")
                for index, config in enumerate(cases, start=1)
            ],
        }
        manager._write_continuous_manifest(manifest_path, manifest)
        for index, config in enumerate(cases, start=1):
            manager._write_continuous_experiment_info(index, config, len(cases))
        manager.state["status"] = "running"
        manager.state["continuous_experiment"] = {
            "batch_id": batch_id,
            "status": "running",
            "current_case": 0,
            "total_cases": CONTINUOUS_CASE_COUNT,
            "current_case_name": "",
            "manifest_path": str(manifest_path),
            "cases": manifest["cases"],
        }

        started = []

        def finish_case(config, round_id):
            started.append((config["experiment_id"], round_id))
            manager.state.update(status="completed", current_round=round_id)

        base_config = manager.get_config()
        with patch.object(manager, "_start_round", side_effect=finish_case):
            manager._run_continuous_experiments(
                cases, manifest, manifest_path, base_config
            )

        saved = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert saved["status"] == "completed"
        assert [case["status"] for case in saved["cases"]] == [
            "completed"
        ] * CONTINUOUS_CASE_COUNT
        assert started == [(case["experiment_id"], 1) for case in cases]
        assert manager.state["status"] == "completed"
        assert manager.state["continuous_experiment"]["status"] == "completed"
        assert manager.get_config()["experiment_id"] == base_config["experiment_id"]


def test_continuous_start_rejects_missing_matrix_cases():
    with tempfile.TemporaryDirectory() as temporary_directory:
        manager = ExperimentManager(
            ProcessManager(), Path(temporary_directory), client_count=10
        )
        manager.config["experiment_type"] = "federated"
        with pytest.raises(ValueError, match="24개"):
            manager._prepare_continuous_cases(
                scenario_cases()[:-1], "batch-test", manager.get_config()
            )
        invalid = scenario_cases()
        invalid[0]["delay_clients"] = [True]
        with pytest.raises(ValueError, match="지연 클라이언트"):
            manager._prepare_continuous_cases(
                invalid, "batch-test", manager.get_config()
            )


def test_active_continuous_batch_blocks_other_experiment_operations():
    with tempfile.TemporaryDirectory() as temporary_directory:
        manager = ExperimentManager(
            ProcessManager(), Path(temporary_directory), client_count=10
        )
        manager.state["status"] = "completed"
        manager.state["continuous_experiment"] = {"status": "running"}

        assert manager.status()["running"] is True
        with pytest.raises(ValueError, match="실행 중"):
            manager.update_config({"learning_rate": 0.02})
        with pytest.raises(RuntimeError, match="연속 실험"):
            manager.start()
        with pytest.raises(RuntimeError, match="실행 중"):
            manager.reset()


def test_manager_page_exposes_continuous_experiment_controls():
    app = create_app(
        "127.0.0.1",
        0,
        client_count=10,
        server_port=5000,
        client_base_port=5101,
    )
    response = app.test_client().get("/")
    assert response.status_code == 200
    page = response.get_data(as_text=True)
    assert 'id="continuous-mode-panel"' in page
    assert 'id="continuous-experiments"' in page
    assert 'id="start-continuous-button"' in page
    assert "continuous-experiments.js" in page
    assert any(
        rule.rule == "/api/experiment/continuous/start"
        for rule in app.url_map.iter_rules()
    )

