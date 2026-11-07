import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from fedsec.__main__ import main
from fedsec.config import (PROJECT_ROOT, EvaluationConfig,
                           ExperimentConfig, TrainingConfig, load_config)
from fedsec.data import make_iid_partition, validate_partition
from fedsec.simulation import participation_schedule


class ConfigurationTests(unittest.TestCase):
    def test_presets_and_inheritance(self):
        base = load_config(PROJECT_ROOT / "configs/experiment.json")
        development = load_config(PROJECT_ROOT / "configs/development.json")
        self.assertEqual(base.num_clients, 10)
        self.assertEqual(base.data.validation_size, 0)
        self.assertEqual(development.num_rounds, 1)
        self.assertEqual(development.training.local_epochs, 1)
        self.assertEqual(development.training.batch_size, base.training.batch_size)
        self.assertEqual(development.evaluation.split, "validation")

    def test_unknown_keys_cycles_and_bad_values(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            for content in ({"training": {"learning_rtae": 0.1}}, {"extends": "config.json"},
                            {"participation_rate": 0}, {"num_rounds": True},
                            {"training": {"learning_rate": float("nan")}},
                            {"runtime": {"device": "unknown"}}):
                path.write_text(json.dumps(content), encoding="utf-8")
                with self.subTest(content=content), self.assertRaises(ValueError):
                    load_config(path)

    def test_ambiguous_json_and_nonfinite_parameters_are_rejected(self):
        invalid = (
            '{"num_rounds": 1, "num_rounds": 20}',
            '{"training": {"learning_rate": 0.01, "learning_rate": 0.1}}',
            '{"defense": {"parameters": {"max_norm": NaN}}}',
            '{"defense": {"parameters": {"max_norm": 1e400}}}',
            '{"data": {"partition_file": ""}}',
            '{"initial_checkpoint": " "}',
            '{"participation_schedule": {"1": [[0]]}}',
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            for text in invalid:
                path.write_text(text, encoding="utf-8")
                with self.subTest(text=text), self.assertRaises(ValueError):
                    load_config(path)

    def test_participation_is_reproducible_and_explicit_schedule_wins(self):
        config = ExperimentConfig(participation_rate=0.5, participation_schedule={"1": [0, 2, 7]})
        first = participation_schedule(config)
        self.assertEqual(first, participation_schedule(config))
        self.assertEqual(first["1"], [0, 2, 7])
        self.assertEqual(len(first["2"]), 5)

    def test_partition_balanced_disjoint_and_reproducible(self):
        targets = [label for label in range(10) for _ in range(50)]
        arguments = dict(num_clients=10, validation_size=100, seed=42)
        partition = make_iid_partition(targets, **arguments)
        validate_partition(partition, targets, num_clients=10)
        self.assertEqual(partition, make_iid_partition(targets, **arguments))
        for indices in partition["client_indices"].values():
            self.assertEqual(len(indices), 40)
            self.assertEqual({label: sum(targets[i] == label for i in indices) for label in range(10)},
                             {label: 4 for label in range(10)})
        partition["client_indices"]["0"][0] = partition["validation_indices"][0]
        with self.assertRaises(ValueError):
            validate_partition(partition, targets, num_clients=10)

    def test_nondivisible_client_count_is_supported(self):
        targets = [label for label in range(10) for _ in range(11)]
        partition = make_iid_partition(targets, num_clients=3, validation_size=13, seed=1)
        self.assertEqual(len(partition["validation_indices"]), 13)
        self.assertLessEqual(max(map(len, partition["client_indices"].values())) - min(map(len, partition["client_indices"].values())), 1)

    def test_validate_and_dry_run_never_load_data_or_train(self):
        config = str(PROJECT_ROOT / "configs/development.json")
        with patch("fedsec.simulation.simulate", side_effect=AssertionError("simulation forbidden")), \
                patch("fedsec.simulation.load_cifar10", side_effect=AssertionError("dataset forbidden")), \
                patch("fedsec.training.train_local", side_effect=AssertionError("training forbidden")), \
                contextlib.redirect_stdout(io.StringIO()):
            main(["validate", "--config", config, "--mode", "compare"])
            main(["simulate", "--config", config, "--mode", "compare", "--dry-run"])

    def test_validation_requires_holdout_and_attack_ids_must_exist(self):
        with self.assertRaises(ValueError):
            ExperimentConfig(evaluation=EvaluationConfig(split="validation"))
        with self.assertRaises(ValueError):
            ExperimentConfig(num_clients=2)
        with self.assertRaises(ValueError):
            TrainingConfig(learning_rate=-1)


if __name__ == "__main__":
    unittest.main()
