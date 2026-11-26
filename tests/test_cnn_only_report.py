import io
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from openpyxl import load_workbook

import run_all_trials as runner
from experiments.cnn_report import load_cnn_workbook, save_cnn_workbook


class CnnReportTest(unittest.TestCase):
    def test_divergence_resumes_from_checkpoint_and_is_recorded_once(self):
        import pickle
        import numpy as np
        import torch
        from fedsec.__main__ import _run_config
        from fedsec.config import ExperimentSettings
        from fedsec.training import NonFiniteLossError

        with TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            root = Path(directory)
            (root / "batch").write_bytes(pickle.dumps({"data": np.zeros((3, 3072), dtype=np.uint8), "labels": [0, 1, 2]}))
            config = runner.cnn_model_plans("cpu")[0][0]
            config["experiment"]["num_rounds"] = 2
            config["datasets"] = {"source_dir": directory,
                                  "clients": [{"client_id": 0, "ranges": [{"file": "batch", "start": 0, "end": 2}]}],
                                  "test_ranges": [{"file": "batch", "start": 2, "end": 3}]}
            settings = ExperimentSettings.from_dict(config["experiment"])
            trial_dir = root / config["experiment_id"]
            phase = trial_dir / "normal"
            cross_entropy = torch.nn.functional.cross_entropy
            training_batches = 0

            def diverge_on_second_round(*args, **kwargs):
                nonlocal training_batches
                if torch.is_grad_enabled():
                    training_batches += 1
                    if training_batches >= 2:
                        return torch.tensor(float("nan"))
                return cross_entropy(*args, **kwargs)

            with patch("torch.nn.functional.cross_entropy", side_effect=diverge_on_second_round):
                with self.assertRaises(NonFiniteLossError):
                    _run_config(config, settings, trial_dir, resume=True)
                self.assertTrue((phase / "round_checkpoint.pt").is_file())
                self.assertFalse((phase / "failure.json").exists())
                _run_config(config, settings, trial_dir, resume=True, record_trial_failure=True)
            failure = runner.read_json(phase / "failure.json")
            self.assertEqual(failure["last_completed_round"], 1)
            self.assertIn("round=2, client=0, epoch=1, batch=1", failure["reason"])
            with patch("fedsec.__main__._train_and_save", side_effect=AssertionError("unexpected retry")):
                _run_config(config, settings, trial_dir, resume=True, record_trial_failure=True)
                changed = runner.cnn_candidate_config(config, config["experiment_id"], {"num_rounds": 3})
                with self.assertRaises(ValueError):
                    _run_config(changed, ExperimentSettings.from_dict(changed["experiment"]), trial_dir,
                                resume=True, record_trial_failure=True)
            self.assertEqual(runner.run_cnn_trial(root, config).failure, failure)

    def test_validation_selection_and_official_report(self):
        self.check_report(set())

    def test_failed_trials_are_reported_and_other_trials_continue(self):
        self.check_report({("E03_smallcnn", "E03_smallcnn"), ("E04_cnn4", "E13_lr0p10"),
                           ("E04_cnn4", "E19_crop"), ("E04_cnn4", "E33_selected_normal_seed202")})

    def check_report(self, failures):
        plans = runner.cnn_model_plans("cuda")
        validation_runs = []
        final_runs = []

        def trial(model_dir, config):
            model = model_dir.name
            trial_id = config["experiment_id"]
            settings = config["experiment"]
            self.assertEqual(settings["training"]["device"], "cuda")
            path = runner.cnn_report_path(model_dir, trial_id)
            path.parent.mkdir(parents=True, exist_ok=True)
            config_path = path.parents[1] / "config.json"
            runner.write_json(config_path, config)
            if "final_test_ranges" in config["datasets"]:
                self.assertEqual(len(validation_runs), 38)
                frozen = runner.read_json(model_dir.parent / "frozen_selections.json")["models"]
                self.assertEqual(set(frozen), {"E03_smallcnn", "E04_cnn4"})
                self.assertIn(config, list(runner.cnn_final_configs(frozen[model]["config"])))
                seed = settings["seed"]
                final_runs.append((model, seed))
                accuracy = {101: 0.69, 202: 0.72, 303: 0.75}[seed] + (0.1 if model == "E04_cnn4" else 0)
                if (model, trial_id) not in failures:
                    runner.write_json(path, {"final_test": {"split": "test", "accuracy": accuracy,
                                                             "loss": 1.0, "num_test_samples": 10000}})
            else:
                validation_runs.append(config)
                scores = ({"E07_lr0p02": 0.61, "E15_dropout0p2": 0.70, "E18_flip": 0.74}
                          if model == "E03_smallcnn" else
                          {"E12_lr0p05": 0.73, "E16_dropout0p3": 0.82, "E18_flip": 0.80})
                accuracy = scores.get(trial_id, 0.50)
                if trial_id == "E18_flip":
                    expected = (0.02, 0.2, 50) if model == "E03_smallcnn" else (0.05, 0.3, 50)
                    self.assertEqual((settings["training"]["learning_rate"], settings["model_config"]["dropout"],
                                      settings["num_rounds"]), expected)
            if (model, trial_id) in failures:
                return runner.FailedTrial(trial_id, config_path, {"reason": "Training loss is NaN/Infinity",
                                                                  "last_completed_round": 20, "experiment_id": trial_id})
            return runner.TrialResult(trial_id, config_path, accuracy, 1.0, settings["model_config"]["dropout"])

        with TemporaryDirectory() as directory, redirect_stdout(io.StringIO()), patch.object(runner, "run_cnn_trial", trial):
            target = Path(directory)
            summaries = runner.run_cnn_suite(target, plans)
            expected_runs = [(model, seed) for model in summaries for seed in runner.FINAL_SEEDS]
            self.assertEqual(final_runs, expected_runs)
            for model, expected_baseline, delta in [("E03_smallcnn", "E15_dropout0p2", 4),
                                                     ("E04_cnn4", "E16_dropout0p3", -2)]:
                aug = summaries[model]["stages"]["augmentation"]
                self.assertEqual(aug["baseline_experiment_id"], expected_baseline)
                flip = next(candidate for candidate in aug["candidates"] if candidate["experiment_id"] == "E18_flip")
                self.assertAlmostEqual(flip["accuracy_delta_pp"], delta)
            self.assertAlmostEqual(summaries["E03_smallcnn"]["mean_test_accuracy"], 0.72)
            if failures:
                self.assertIsNone(summaries["E04_cnn4"]["mean_test_accuracy"])
                crop = next(c for c in summaries["E04_cnn4"]["stages"]["augmentation"]["candidates"]
                            if c["experiment_id"] == "E19_crop")
                self.assertIsNone(crop["accuracy_delta_pp"])
            else:
                self.assertAlmostEqual(summaries["E04_cnn4"]["mean_test_accuracy"], 0.82)
            with patch.object(runner, "tune_cnn_models", side_effect=AssertionError("unexpected retuning")):
                self.assertEqual(runner.run_cnn_suite(target, plans), summaries)
                with self.assertRaises(ValueError):
                    runner.run_cnn_suite(target, runner.cnn_model_plans("cpu"))

            path = save_cnn_workbook(load_cnn_workbook(), summaries, runner.STAGE_LABELS, target)
            book = load_workbook(path)
            final = book["최종 비교"]
            for number, (model, seed) in enumerate(expected_runs, 7):
                self.assertEqual(final[f"B{number}"].value, seed)
                result = next(item for item in summaries[model]["official_tests"] if item["seed"] == seed)
                if result.get("status") == "failed":
                    self.assertIsNone(final[f"E{number}"].value)
                    self.assertIn("NaN/Infinity", final[f"G{number}"].value)
                else:
                    self.assertAlmostEqual(final[f"E{number}"].value, result["accuracy"])
            if failures:
                validation = book["검증 기록"]
                self.assertEqual(validation["C30"].value, "E13_lr0p10")
                self.assertIsNone(validation["E30"].value)
                self.assertIn("실패", validation["G30"].value)
            self.assertEqual([final["C22"].value, final["C23"].value], ["flip", "none"])
            for address, expected in [("F8", "$D$7"), ("F14", "$D$13")]:
                self.assertIn(expected, book["증강 비교"][address].value)
            self.assertIn("AVERAGE(E7:E9)", final["E16"].value)
            self.assertIn("AVERAGE(E10:E12)", final["E17"].value)
            book.close()
