"""Workflow tests replace ALL training and evaluation with fixed-tensor stubs."""

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch
from torch.utils.data import TensorDataset
from fedsec.artifacts import load_checkpoint, save_checkpoint
from fedsec.config import (AttackConfig, DataConfig, DefenseConfig, EvaluationConfig,
                           ExperimentConfig, ModelConfig, RuntimeConfig, TrainingConfig)
from fedsec.contracts import ClientUpdate
from fedsec.integration import train_from_checkpoint
from fedsec.model import SmallCNN, build_model
from fedsec.simulation import simulate
from fedsec.tensors import clone_state, state_hash
from fedsec.training import LocalTrainingResult


class WorkflowTests(unittest.TestCase):
    def test_existing_output_is_rejected_before_loading_data(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "existing_run"
            output.mkdir()
            marker = output / "original.txt"
            marker.write_text("keep", encoding="utf-8")
            config = ExperimentConfig(runtime=RuntimeConfig(output_root=directory))
            with patch("fedsec.simulation.load_cifar10", side_effect=AssertionError("dataset forbidden")), \
                    patch("fedsec.simulation.build_model", side_effect=AssertionError("model forbidden")):
                with self.assertRaises(FileExistsError):
                    simulate(config, mode="normal", run_name="existing_run")
            self.assertEqual(marker.read_text(encoding="utf-8"), "keep")

    def test_compare_pipeline_with_stubs_and_no_model_execution(self):
        targets = [label for label in range(10) for _ in range(5)]
        source = TensorDataset(torch.zeros(50, 3, 32, 32), torch.tensor(targets))
        source.targets = targets
        calls = []

        def fake_train(state, dataset, *, client_id, context, training_config, **kwargs):
            self.assertEqual(training_config, config.training)
            self.assertNotIn("request", kwargs)
            self.assertNotIn("optimizer_config", kwargs)
            calls.append((context.round_id, client_id, state_hash(state), tuple(dataset.indices)))
            delta = {name: torch.full_like(tensor, 0.001 * (client_id + 1)) for name, tensor in state.items()}
            update = ClientUpdate(context.experiment_id, client_id, context.round_id,
                                  context.model_version, len(dataset), state_hash(state), delta)
            kwargs["on_epoch"]({"epoch": 1, "loss": 1.0, "accuracy": 0.0})
            kwargs["on_epoch"]({"phase": "local_complete", "update_norm": 0.0})
            return LocalTrainingResult(update, [{"epoch": 1, "loss": 1.0, "accuracy": 0.0}])

        def fake_evaluate(model, loader, device):
            return {"accuracy": 0.25, "loss": 1.0, "num_samples": len(loader.dataset)}

        with tempfile.TemporaryDirectory() as directory:
            config = ExperimentConfig(num_clients=3, num_rounds=2,
                model=ModelConfig(channels=(2, 3), hidden_size=4),
                training=TrainingConfig(local_epochs=1),
                data=DataConfig(validation_size=10), evaluation=EvaluationConfig(split="validation"),
                attack=AttackConfig(malicious_clients=(0, 2)),
                runtime=RuntimeConfig(output_root=directory))
            with patch("fedsec.simulation.load_cifar10", return_value=source), \
                    patch.object(SmallCNN, "forward", side_effect=AssertionError("forward forbidden")), \
                    patch("torch.Tensor.backward", side_effect=AssertionError("backward forbidden")), \
                    patch("torch.optim.SGD", side_effect=AssertionError("optimizer forbidden")), \
                    contextlib.redirect_stdout(io.StringIO()):
                output = simulate(config, mode="compare", run_name="stub_only",
                                  train_fn=fake_train, evaluate_fn=fake_evaluate)
            comparison = json.loads((output / "comparison.json").read_text(encoding="utf-8"))
            self.assertTrue(comparison["shared_initial_model"])
            self.assertEqual(comparison["accuracy_drop"], 0.0)
            normal = load_checkpoint(output / "normal/round_0002/global.pt")
            attack = load_checkpoint(output / "attack/round_0002/global.pt")
            self.assertNotEqual(normal["model_hash"], attack["model_hash"])
            self.assertEqual(normal["model_version"], 2)
            initial = load_checkpoint(output / "initial.pt")["model_state"]
            # 40 samples are assigned as 14/13/13. Two fixed-update rounds:
            # normal: (14*.001 + 13*.002 + 13*.003)/40 = .001975
            # attack: (14*(-.01) + 13*.002 + 13*(-.01))/40 = -.0061
            for name, value in initial.items():
                torch.testing.assert_close(normal["model_state"][name], value + 2 * 0.001975, atol=1e-7, rtol=1e-5)
                torch.testing.assert_close(attack["model_state"][name], value - 2 * 0.0061, atol=1e-7, rtol=1e-5)
            self.assertEqual(calls[:3], calls[6:9])  # same first-round W/data/IDs
            self.assertTrue((output / "attack/round_0001/submitted_updates/client_0.pt").exists())
            self.assertTrue((output / "normal/metrics.csv").exists())
            with patch("fedsec.simulation.load_cifar10", return_value=source):
                with self.assertRaises(FileExistsError):
                    simulate(config, mode="normal", run_name="stub_only", train_fn=fake_train, evaluate_fn=fake_evaluate)

    def test_defense_without_implementation_fails_before_data_access(self):
        config = ExperimentConfig(defense=DefenseConfig(enabled=True, type="norm_clipping"))
        with patch("fedsec.simulation.load_cifar10", side_effect=AssertionError("dataset forbidden")):
            with self.assertRaises(ValueError):
                simulate(config, mode="normal")

    def test_checkpoint_and_api_bridge_validate_before_calling_training(self):
        model_config = ModelConfig(channels=(2, 3), hidden_size=4)
        state = clone_state(build_model(model_config).state_dict())
        data_config = DataConfig()
        payload = {"round_id": 1, "model_version": 4, "local_epochs": 2,
                   "batch_size": 5, "learning_rate": 0.02}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "global.pt"
            save_checkpoint(path, state=state, model_config=model_config, data_config=data_config,
                            experiment_id="exp", model_version=4)
            checkpoint = load_checkpoint(path)
            self.assertEqual(checkpoint["model_hash"], state_hash(state))
            dataset = object()
            defaults = TrainingConfig(local_epochs=7, batch_size=64, learning_rate=0.1,
                                      momentum=0.9, weight_decay=0.0001)
            kwargs = dict(client_id=2, experiment_id="exp", data_config=data_config,
                          training_defaults=defaults, runtime=RuntimeConfig(), seed=42)
            sentinel = object()
            with patch("fedsec.integration.train_local", return_value=sentinel) as mocked:
                result = train_from_checkpoint(path, payload, dataset, **kwargs)
                self.assertIs(result, sentinel)
                effective = mocked.call_args.kwargs["training_config"]
                self.assertEqual((effective.local_epochs, effective.batch_size, effective.learning_rate), (2, 5, 0.02))
                self.assertEqual((effective.optimizer, effective.momentum, effective.weight_decay), ("sgd", 0.9, 0.0001))
                self.assertEqual((defaults.local_epochs, defaults.batch_size, defaults.learning_rate), (7, 64, 0.1))
                context = mocked.call_args.kwargs["context"]
                self.assertEqual((context.experiment_id, context.round_id, context.model_version, context.selected_clients),
                                 ("exp", 1, 4, (2,)))
                self.assertNotIn("request", mocked.call_args.kwargs)
                self.assertNotIn("optimizer_config", mocked.call_args.kwargs)
                with self.assertRaises(ValueError):
                    train_from_checkpoint(path, {**payload, "model_version": 5}, dataset, **kwargs)
                with self.assertRaises(ValueError):
                    train_from_checkpoint(path, payload, dataset,
                        **{**kwargs, "data_config": DataConfig(normalization_std=(1.0, 1.0, 1.0))})
                self.assertEqual(mocked.call_count, 1)
            checkpoint["model_state"][next(iter(state))].zero_()
            torch.save(checkpoint, path)
            with self.assertRaises(ValueError):
                load_checkpoint(path)


if __name__ == "__main__":
    unittest.main()
