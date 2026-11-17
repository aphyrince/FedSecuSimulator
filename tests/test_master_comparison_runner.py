"""통합 실행의 순서/중단/비교 조건을 검증한다. 네트워크와 실제 학습은 사용하지 않는다."""

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from common.training import TrainingSettings
from exp_manager.services.comparison_runner import run_comparison_experiment, run_federated_learning


class FakeManagerAPI:
    def __init__(self):
        self.config = {**TrainingSettings().to_dict(), "experiment_id": "ui",
                       "experiment_type": "federated", "seed": 42, "num_rounds": 2,
                       "round_timeout": 30, "model_config": {}, "malicious_clients": [0, 1],
                       "attack_scale": -5.0, "participation_rate": 1.0,
                       "defense_enabled": True, "defense": "coordinate_median",
                       "round_configs": [{"attack_enabled": True, "defense_enabled": True,
                                          "learning_rate": 0.02}]}
        self.starts = []
        self.state = "idle"
        self.fail_normal = False
        self.unfair = False
        self.hang = False
        self.stop_requests = 0

    def status(self):
        return {"status": self.state, "config": deepcopy(self.config),
                "round_results": self.records() if self.state == "completed" else [],
                "result_path": f"results/{self.config['experiment_id']}/results.json"}

    def get_json(self, url, **kwargs):
        if url.endswith("/config"):
            return deepcopy(self.config)
        if self.state == "running" and not self.hang:
            self.state = "failed" if self.fail_normal and not self.config["attack_enabled"] else "completed"
        return self.status()

    def post_json(self, url, payload, **kwargs):
        if url.endswith("/config"):
            self.config = deepcopy(payload)
            return deepcopy(payload)
        if url.endswith("/start"):
            self.starts.append(deepcopy(self.config))
            self.state = "running"
        elif url.endswith("/stop"):
            self.state = "stopping"
            self.stop_requests += 1
        return self.status()

    def records(self):
        attack = self.config["attack_enabled"]
        return [{"round": i, "initial_model_hash": "initial",
                 "training_config": TrainingSettings.from_dict(self.config).to_dict(),
                 "training_datasets": {"0": {"fingerprint": "train", "num_samples": 5}},
                 "selected_clients": [0, 1],
                 "accepted_clients": [0] if attack and self.unfair else [0, 1],
                 "evaluation": {"dataset_fingerprint": "test", "num_test_samples": 10},
                 "accuracy": 0.4 if attack else 0.7, "loss": 1.0,
                 "attack_enabled": attack, "attack_applied": attack}
                for i in range(1, self.config["num_rounds"] + 1)]


class ComparisonRunnerTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.output = Path(directory.name)
        self.api = FakeManagerAPI()
        for target, kwargs in (
                ("exp_manager.services.comparison_runner.RestClient", {"return_value": self.api}),
                ("exp_manager.services.comparison_runner.time.sleep", {"return_value": None}),
                ("common.model.SmallCNN.forward", {"side_effect": AssertionError("No real inference")}),
                ("torch.Tensor.backward", {"side_effect": AssertionError("No real training")})):
            mock = patch(target, **kwargs)
            mock.start()
            self.addCleanup(mock.stop)

    def run_pair(self):
        return run_comparison_experiment({"experiment_id": "pair"}, output_dir=self.output)

    def test_runs_both_conditions_with_same_snapshot_and_saves_real_comparison(self):
        original = deepcopy(self.api.config)
        result = self.run_pair()
        normal, attack = self.api.starts
        self.assertEqual([c["experiment_id"] for c in self.api.starts], ["pair_normal", "pair_attack"])
        for config, enabled in ((normal, False), (attack, True)):
            self.assertEqual(config["attack_enabled"], enabled)
            self.assertFalse(config["defense_enabled"])
            self.assertEqual(config["defense"], "none")
            self.assertEqual(config["round_configs"][0]["attack_enabled"], enabled)
            self.assertFalse(config["round_configs"][0]["defense_enabled"])
            self.assertEqual(config["round_configs"][0]["learning_rate"], 0.02)
        for key in ("seed", "model_config", "local_epochs", "malicious_clients"):
            self.assertEqual(normal[key], attack[key])
            self.assertEqual(normal[key], original[key])
        self.assertAlmostEqual(result["final"]["accuracy_drop_percentage_points"], 30)
        self.assertEqual(json.loads((self.output / "pair_comparison.json").read_text(encoding="utf-8")), result)

    def test_baseline_failure_does_not_start_attack(self):
        self.api.fail_normal = True
        with self.assertRaisesRegex(RuntimeError, "failed"):
            self.run_pair()
        self.assertEqual(len(self.api.starts), 1)
        self.assertFalse(list(self.output.glob("*.json")))

    def test_unfair_results_are_not_saved(self):
        self.api.unfair = True
        with self.assertRaisesRegex(ValueError, "accepted_clients"):
            self.run_pair()
        self.assertFalse(list(self.output.glob("*.json")))

    def test_existing_comparison_cannot_be_overwritten(self):
        self.run_pair()
        with self.assertRaises(FileExistsError):
            self.run_pair()
        self.assertEqual(len(self.api.starts), 2)

    def test_busy_manager_is_not_reconfigured(self):
        self.api.state = "running"
        self.api.hang = True
        with self.assertRaisesRegex(RuntimeError, "실행 중"):
            self.run_pair()
        self.assertFalse(self.api.starts)
        self.assertEqual(self.api.config["experiment_id"], "ui")

    def test_timeout_requests_stop_and_does_not_start_attack(self):
        self.api.hang = True
        with patch("exp_manager.services.comparison_runner.time.monotonic", side_effect=[0, 2]):
            with self.assertRaises(TimeoutError):
                run_comparison_experiment({"experiment_id": "pair"}, output_dir=self.output, timeout=1)
        self.assertEqual(len(self.api.starts), 1)
        self.assertEqual(self.api.stop_requests, 1)

    def test_incomplete_rounds_rejected(self):
        config = {**self.api.config, "attack_enabled": False}
        with patch.object(self.api, "records", return_value=[{"round": 1}]):
            with self.assertRaisesRegex(RuntimeError, "누락"):
                run_federated_learning(config)


if __name__ == "__main__":
    unittest.main()
