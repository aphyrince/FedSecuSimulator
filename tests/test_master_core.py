"""1번 담당 핵심 계산과 REST 연결의 일치 여부 및 독립 실행 흐름을 검증한다. 실제 학습은 실행하지 않는다."""

import ast
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch.utils.data import TensorDataset

from fedsec import ExperimentSettings, TrainingSettings, ClientUpdate, run_comparison_experiment
from fedsec.aggregation import aggregate_fedavg
from fedsec.attack import collude_updates
from fedsec.contracts import LocalTrainingResult, EvaluationResult
from fedsec.data import load_datasets
from fedsec.tensors import copy_state
from common.model import create_model, decode_weights, weights_hash
from clients.services.local_training import train_local_model as train_adapter
from test_master_algorithms import make_update, vector_model
from server.services.fedavg import aggregate_fedavg as aggregate_adapter


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.tensor([10.0]))

    def forward(self, _):
        raise AssertionError("No actual model execution")


class CoreTests(unittest.TestCase):
    def setUp(self):
        for target in ("torch.Tensor.backward", "torch.optim.SGD", "torch.optim.Adam", "fedsec.model.SmallCNN.forward"):
            guard = patch(target, side_effect=AssertionError("No actual training/inference"))
            guard.start()
            self.addCleanup(guard.stop)

    def test_core_has_no_api_or_application_dependencies(self):
        forbidden = {"common", "clients", "server", "exp_manager", "flask", "requests", "urllib"}
        root = Path(__file__).resolve().parents[1] / "fedsec"
        for path in root.glob("*.py"):
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
                names = [node.module] if isinstance(node, ast.ImportFrom) else (
                    [item.name for item in node.names] if isinstance(node, ast.Import) else [])
                for name in names:
                    self.assertNotIn((name or "").split(".")[0], forbidden, str(path))

    def test_fedavg_adapter_uses_identical_core_math(self):
        model = vector_model()
        payloads = [make_update(model, 0, {"w": [1.0, 2.0]}, 2),
                    make_update(model, 1, {"w": [3.0, 4.0]}, 6)]
        updates = [ClientUpdate(p["client_id"], p["num_samples"], decode_weights(p["update"]["values"]))
                   for p in payloads]
        before = deepcopy(model)
        expected = aggregate_fedavg(decode_weights(model["weights"]), updates)
        actual = decode_weights(aggregate_adapter(model, payloads)["global_model"]["weights"])
        torch.testing.assert_close(expected["w"], actual["w"])
        self.assertEqual(model, before)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            aggregate_fedavg(decode_weights(model["weights"]), [updates[0], updates[0]])

    def test_collusion_keeps_clean_inputs_and_validates_shapes(self):
        left, right = {"w": torch.tensor([1.0])}, {"w": torch.tensor([3.0])}
        self.assertEqual(collude_updates([left, right])["w"].item(), -10)
        self.assertEqual(left["w"].item(), 1)
        with self.assertRaises(ValueError):
            collude_updates([left])
        with self.assertRaises(ValueError):
            collude_updates([left, {"w": torch.tensor([2.0, 3.0])}])

    def test_two_round_comparison_resets_initial_model_and_colludes_clean_updates(self):
        clients = {0: TensorDataset(torch.zeros(2, 1)), 1: TensorDataset(torch.zeros(6, 1))}
        for key, dataset in clients.items():
            dataset.fingerprint = f"train-{key}"
        test = TensorDataset(torch.zeros(3, 1))
        test.fingerprint = "test"
        starts, events = [], []

        def fake_train(model, dataset, settings, *, seed, round_id, client_id):
            starts.append((round_id, client_id, model.w.item(), seed))
            delta = {"w": torch.tensor([1.0 if client_id == 0 else 3.0])}
            weights = {"w": model.w.detach().clone() + delta["w"]}
            return LocalTrainingResult(weights, delta, len(dataset), [{"loss": 1.0, "accuracy": 0.5}])

        def fake_evaluate(model, dataset, settings, *, seed):
            return EvaluationResult(0.7 if model.w.item() > 0 else 0.4, 1.0, len(dataset), 1)

        with patch("fedsec.experiment.create_model", side_effect=lambda *args: TinyModel()), \
                patch("fedsec.experiment.train_local_model", side_effect=fake_train), \
                patch("fedsec.experiment.evaluate_global_model", side_effect=fake_evaluate):
            result = run_comparison_experiment(ExperimentSettings(num_rounds=2), clients, test, on_round=events.append)
        self.assertEqual([item[2] for item in starts], [10, 10, 12.5, 12.5, 10, 10, 0, 0])
        self.assertEqual(result.normal.final_weights["w"].item(), 15)
        self.assertEqual(result.attacked.final_weights["w"].item(), -10)
        self.assertEqual(len(events), 4)
        self.assertAlmostEqual(result.comparison["final"]["accuracy_drop_percentage_points"], 30)
        json.dumps(result.normal.report())

    def test_invalid_attack_fails_before_normal_training(self):
        data = TensorDataset(torch.zeros(1, 1))
        data.fingerprint = "data"
        with patch("fedsec.experiment.train_local_model") as train:
            with self.assertRaises(ValueError):
                run_comparison_experiment(ExperimentSettings(), {0: data}, data)
            train.assert_not_called()

    def test_training_adapter_converts_payload_and_preserves_deadline(self):
        payload = create_model(42, {"channels": [2, 3], "hidden_size": 4})
        payload["update_deadline"] = "2000-01-01T00:00:00+00:00"
        weights = decode_weights(payload["weights"])
        delta = {key: torch.zeros_like(value) for key, value in weights.items()}
        dataset = TensorDataset(torch.zeros(2, 1))
        dataset.fingerprint = "train"

        def fake_train(model, data, settings, **kwargs):
            self.assertIs(data, dataset)
            self.assertEqual((kwargs["round_id"], kwargs["client_id"]), (1, 0))
            with self.assertRaises(TimeoutError):
                kwargs["check_cancelled"]()
            return LocalTrainingResult(weights, delta, 2, [{"loss": 0.5, "accuracy": 1.0}])

        with patch("clients.services.local_training.CifarRangeDataset", return_value=dataset), \
                patch("clients.services.local_training.train_core", side_effect=fake_train):
            result = train_adapter(payload, {}, {}, 1, 0)
        self.assertEqual(result["num_samples"], 2)
        self.assertEqual(result["base_model_hash"], weights_hash(weights))
        self.assertNotIn("update_deadline", result["local_model"])
        json.dumps(result)

    def test_manifest_rejects_leakage_before_reading_data(self):
        manifest = {"source_dir": ".", "clients": [{"client_id": 0, "ranges": [{"file": "absent", "start": 0, "end": 5}]}],
                    "test_ranges": [{"file": "absent", "start": 4, "end": 6}]}
        with self.assertRaisesRegex(ValueError, "overlap"):
            load_datasets(manifest, TrainingSettings())

    def test_unknown_config_does_not_silently_change_experiment(self):
        for values in ({"participation_rate": 0.5}, {"training": {"lr": 0.2}}):
            with self.assertRaisesRegex(ValueError, "Unknown"):
                ExperimentSettings.from_dict(values)

    def test_cli_saves_reports_and_compatible_checkpoints_without_training(self):
        from fedsec.__main__ import main
        from common.model import load_model_checkpoint
        dataset = TensorDataset(torch.zeros(2, 1))
        dataset.fingerprint = "fixture"

        def fake_train(model, data, settings, **kwargs):
            weights = copy_state(model.state_dict())
            delta = {key: torch.zeros_like(value) for key, value in weights.items()}
            return LocalTrainingResult(weights, delta, len(data), [{"loss": 1.0, "accuracy": 0.5}])

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = root / "config.json"
            config.write_text(json.dumps({"experiment_id": "test", "datasets": {},
                                         "experiment": {"num_rounds": 1,
                                                        "model_config": {"channels": [2, 3], "hidden_size": 4}}}))
            with patch("sys.argv", ["fedsec", "--config", str(config), "--output-dir", str(root)]), \
                    patch("fedsec.__main__.load_datasets", return_value=({0: dataset, 1: dataset}, dataset)), \
                    patch("fedsec.experiment.train_local_model", side_effect=fake_train), \
                    patch("fedsec.experiment.evaluate_global_model", return_value=EvaluationResult(0.5, 1.0, 2, 1)), \
                    patch("builtins.print"):
                main()
                saved = json.loads((root / "test/comparison.json").read_text(encoding="utf-8"))
                self.assertEqual(saved["final"]["accuracy_drop"], 0)
                for phase in ("normal", "attack"):
                    checkpoint = load_model_checkpoint(root / "test" / phase / "final_model.pt")
                    self.assertEqual(checkpoint["model_version"], 1)
                    self.assertTrue((root / "test" / phase / "rounds/round_001.json").exists())
                with self.assertRaises(SystemExit) as error:
                    main()
                self.assertEqual(error.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
