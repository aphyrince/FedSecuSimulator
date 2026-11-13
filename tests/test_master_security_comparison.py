import json
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from common.training import TrainingSettings
from exp_manager.services.comparison_runner import run_full_security_comparison


class SecurityComparisonTests(unittest.TestCase):
    def test_computes_all_defense_differences_per_seed_and_mean(self):
        config = {
            **TrainingSettings().to_dict(),
            "experiment_id": "security",
            "experiment_type": "federated",
            "seed": 42,
            "num_rounds": 1,
            "round_timeout": 30,
            "model_config": {"channels": [2, 3], "hidden_size": 4},
            "malicious_clients": [0, 1],
            "participation_rate": 1.0,
            "round_configs": [],
        }
        accuracies = {
            "normal": 0.70,
            "attack": 0.75,
            "normal_defense_norm_clipping": 0.65,
            "attack_defense_norm_clipping": 0.80,
            "normal_defense_coordinate_median": 0.72,
            "attack_defense_coordinate_median": 0.74,
        }
        fake_rest = type(
            "FakeRest",
            (),
            {"get_json": lambda _self, _url: deepcopy(config)},
        )()

        def fake_run(case_config, **_kwargs):
            scenario = case_config["experiment_id"].split("-", 2)[-1]
            seed = case_config["seed"]
            return {
                "config": deepcopy(case_config),
                "result_path": f"results/{case_config['experiment_id']}/results.json",
                "round_results": [
                    {
                        "round": 1,
                        "initial_model_hash": f"model-{seed}",
                        "training_config": TrainingSettings.from_dict(case_config).to_dict(),
                        "training_datasets": {
                            "0": {"fingerprint": "train-0", "num_samples": 5}
                        },
                        "selected_clients": [0, 1, 2],
                        "accepted_clients": [0, 1, 2],
                        "evaluation": {
                            "dataset_fingerprint": "test",
                            "num_test_samples": 10,
                        },
                        "accuracy": accuracies[scenario],
                    }
                ],
            }

        with tempfile.TemporaryDirectory() as temporary:
            with patch(
                "exp_manager.services.comparison_runner.RestClient",
                return_value=fake_rest,
            ), patch(
                "exp_manager.services.comparison_runner.run_federated_learning",
                side_effect=fake_run,
            ):
                report = run_full_security_comparison(
                    {"experiment_id": "security"},
                    [42, 43],
                    output_dir=Path(temporary),
                )

            self.assertEqual(len(report["seeds"]), 2)
            self.assertEqual(len(report["run_result_paths"]), 12)
            for seed_result in report["seeds"]:
                norm = seed_result["defenses"]["norm_clipping"]
                self.assertAlmostEqual(norm["attack_impact_percentage_points"], -5.0)
                self.assertAlmostEqual(norm["defense_improvement_percentage_points"], 5.0)
                self.assertAlmostEqual(norm["normal_defense_loss_percentage_points"], 5.0)
                median = seed_result["defenses"]["coordinate_median"]
                self.assertAlmostEqual(median["defense_improvement_percentage_points"], -1.0)
            self.assertAlmostEqual(
                report["mean"]["norm_clipping"]["attack_impact_percentage_points"],
                -5.0,
            )
            saved = json.loads(
                (Path(temporary) / "security_security_comparison.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(saved, report)


if __name__ == "__main__":
    unittest.main()
