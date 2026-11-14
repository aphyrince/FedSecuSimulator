"""고정 텐서와 작은 로컬 피클 예제로 검증한다. 실제 모델 학습은 실행하지 않는다."""

from copy import deepcopy
import json
from pathlib import Path
import pickle
import tempfile
import unittest

import numpy as np
import torch

from common.comparison import compare_accuracy_results
from common.data import CifarRangeDataset
from common.model import (MODEL_FORMAT, PREPROCESSING, create_model, decode_weights,
                          encode_weights, weights_hash, load_model_checkpoint, save_model_checkpoint)
from common.training import TrainingSettings
from common.updates import update_delta
from clients.services.collusion import collude_updates
from clients.services.local_training import extract_model_update
from server.services.byzantine_defense import apply_norm_clipping
from server.services.fedavg import aggregate_coordinate_median, aggregate_fedavg


def vector_model():
    state = {"w": [10.0, 20.0]}
    return {"architecture": MODEL_FORMAT, "model_config": {"channels": [2, 3], "hidden_size": 4},
            "preprocessing": deepcopy(PREPROCESSING), "model_version": 0,
            "experiment_id": "test", "initial_model_hash": weights_hash(state), "weights": state}


def make_update(model, client_id, delta, samples=2, round_id=1):
    initial = decode_weights(model["weights"])
    values = decode_weights(delta, initial)
    local = deepcopy(model)
    local["weights"] = encode_weights({name: initial[name] + values[name] for name in initial})
    local["model_version"] += 1
    return {"client_id": client_id, "round": round_id, "experiment_id": model.get("experiment_id"),
            "base_model_version": model["model_version"], "base_model_hash": weights_hash(initial),
            "num_samples": samples, "valid": True, "local_model": local,
            "dataset_fingerprint": f"dataset-{client_id}", "update": {"values": encode_weights(values)}}


class AlgorithmsTests(unittest.TestCase):
    def setUp(self):
        self.model = vector_model()

    def test_weighted_fedavg_adds_delta_to_start_and_preserves_inputs(self):
        first = make_update(self.model, 0, {"w": [1.0, 2.0]}, 1)
        second = make_update(self.model, 1, {"w": [3.0, 6.0]}, 3)
        original = deepcopy([self.model, first, second])
        result = aggregate_fedavg(self.model, [first, second])
        self.assertEqual(result["global_model"]["weights"], {"w": [12.5, 25.0]})
        self.assertEqual(result["global_model"]["model_version"], 1)
        self.assertEqual(result["total_samples"], 4)
        self.assertEqual(result["client_weights"], {"0": 0.25, "1": 0.75})
        self.assertEqual([self.model, first, second], original)

    def test_unreceived_clients_are_not_in_fedavg_denominator(self):
        update = make_update(self.model, 0, {"w": [2.0, 3.0]}, 10)
        self.assertEqual(aggregate_fedavg(self.model, [update])["global_model"]["weights"], {"w": [12.0, 23.0]})

    def test_collusion_averages_clean_deltas_not_sample_weights_or_norms(self):
        first = make_update(self.model, 0, {"w": [2.0, 4.0]}, 1)
        second = make_update(self.model, 1, {"w": [4.0, 8.0]}, 9)
        original = deepcopy([first, second])
        attacked = collude_updates(self.model, first, [second], scale=-5)
        self.assertEqual(attacked["update"]["values"], {"w": [-15.0, -30.0]})
        self.assertEqual(attacked["local_model"]["weights"], {"w": [-5.0, -10.0]})
        self.assertEqual(attacked["num_samples"], 1)
        self.assertEqual([first, second], original)
        other = collude_updates(self.model, second, [first], scale=-5)
        self.assertEqual(aggregate_fedavg(self.model, [attacked, other])["global_model"]["weights"], {"w": [-5.0, -10.0]})

    def test_single_malicious_participant_scales_its_own_update(self):
        own = make_update(self.model, 0, {"w": [1.0, 2.0]})
        result = collude_updates(self.model, own, [])
        self.assertTrue(result["update"]["attack_applied"])
        self.assertEqual(
            result["update"]["attack_reason"],
            "single_malicious_client_self_scaled",
        )
        self.assertEqual(result["update"]["values"], {"w": [-5.0, -10.0]})

    def test_clipping_changes_actual_submitted_weights_and_aggregation(self):
        own = make_update(self.model, 0, {"w": [3.0, 4.0]})
        result = apply_norm_clipping([own], 1.0, global_model=self.model)[0]
        torch.testing.assert_close(decode_weights(result["update"]["values"])["w"], torch.tensor([0.6, 0.8]))
        torch.testing.assert_close(decode_weights(aggregate_fedavg(self.model, [result])["global_model"]["weights"])["w"], torch.tensor([10.6, 20.8]))
        self.assertEqual(own["update"]["values"], {"w": [3.0, 4.0]})

    def test_even_coordinate_median_and_sample_weight_difference(self):
        updates = [make_update(self.model, 0, {"w": [2.0, 10.0]}, 1),
                   make_update(self.model, 1, {"w": [4.0, -2.0]}, 99)]
        result = aggregate_coordinate_median(self.model, updates)
        self.assertEqual(result["global_model"]["weights"], {"w": [13.0, 24.0]})

    def test_mismatched_models_shapes_samples_and_duplicate_updates_fail(self):
        valid = make_update(self.model, 0, {"w": [1.0, 2.0]})
        invalid = [dict(valid, base_model_version=99), dict(valid, base_model_hash="other"),
                   dict(valid, experiment_id="other"), dict(valid, num_samples=0), dict(valid, num_samples=True)]
        wrong = deepcopy(valid)
        wrong["update"]["values"] = {"w": [1.0]}
        invalid.append(wrong)
        wrong = deepcopy(valid)
        wrong["local_model"]["weights"] = {"w": [11.0, 999.0]}
        invalid.append(wrong)
        for item in invalid:
            with self.subTest(item=item), self.assertRaises(ValueError):
                aggregate_fedavg(self.model, [item])
        with self.assertRaises(ValueError):
            aggregate_fedavg(self.model, [valid, valid])
        with self.assertRaises(ValueError):
            aggregate_fedavg(self.model, [])

    def test_extraction_uses_final_minus_start(self):
        update = make_update(self.model, 0, {"w": [3.0, 4.0]})
        extracted = extract_model_update(self.model, update["local_model"])
        self.assertEqual(extracted, {"values": {"w": [3.0, 4.0]}, "update_norm": 5.0})

    def test_model_initialization_json_checkpoint_roundtrip_without_forward(self):
        from common.model import SmallCNN
        from unittest.mock import patch
        with patch.object(SmallCNN, "forward", side_effect=AssertionError("forward forbidden")):
            model = create_model(42, {"channels": [2, 3], "hidden_size": 4})
            same = create_model(42, model["model_config"])
            self.assertEqual(model["weights"], same["weights"])
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "model.pt"
                save_model_checkpoint(model, path)
                restored = load_model_checkpoint(path)
                self.assertEqual(restored["weights"], model["weights"])
                self.assertEqual(restored["preprocessing"], PREPROCESSING)
                json.dumps(restored)


class DatasetAndComparisonTests(unittest.TestCase):
    def test_dataset_reads_exact_ranges_labels_and_normalization(self):
        with tempfile.TemporaryDirectory() as directory:
            pixels = np.zeros((6, 3072), dtype=np.uint8)
            pixels[2] = 255
            path = Path(directory) / "batch"
            path.write_bytes(pickle.dumps({b"data": pixels, b"labels": [0, 1, 2, 3, 4, 5]}))
            payload = {"source_dir": directory, "ranges": [{"file": "batch", "start": 2, "end": 5}]}
            dataset = CifarRangeDataset(payload)
            self.assertEqual(len(dataset), 3)
            image, label = dataset[0]
            self.assertEqual((tuple(image.shape), label), ((3, 32, 32), 2))
            torch.testing.assert_close(image, torch.ones(3, 32, 32))
            self.assertEqual(len(CifarRangeDataset(payload, 1)), 1)
            self.assertNotEqual(dataset.fingerprint, CifarRangeDataset(payload, 1).fingerprint)
            duplicated = {**payload, "ranges": payload["ranges"] * 2}
            with self.assertRaises(ValueError):
                CifarRangeDataset(duplicated)
            with self.assertRaises(ValueError):
                CifarRangeDataset({**payload, "ranges": [{"file": "../batch", "start": 0, "end": 1}]})

    def test_invalid_learning_settings_are_rejected(self):
        for values in ({"batch_size": 0}, {"local_epochs": True}, {"learning_rate": float("nan")}, {"device": "unknown"}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                TrainingSettings.from_dict(values)

    def records(self):
        record = {"round": 1, "initial_model_hash": "same", "training_config": {"local_epochs": 2},
                  "training_datasets": {"0": {"fingerprint": "train", "num_samples": 5}},
                  "selected_clients": [0], "accepted_clients": [0],
                  "evaluation": {"dataset_fingerprint": "test", "num_test_samples": 10},
                  "accuracy": 0.7, "loss": 0.8, "attack_enabled": False}
        attack = deepcopy(record)
        attack.update(accuracy=0.45, loss=1.1, attack_enabled=True, attack_applied=True)
        return {"round_results": [record]}, {"round_results": [attack]}

    def test_comparison_reports_percentage_points(self):
        normal, attack = self.records()
        result = compare_accuracy_results(normal, attack)
        self.assertAlmostEqual(result["final"]["accuracy_drop_percentage_points"], 25.0)

    def test_unfair_comparisons_are_rejected(self):
        for key, value in (("initial_model_hash", "different"), ("training_config", {"local_epochs": 5}),
                           ("accepted_clients", [1]), ("evaluation", {"dataset_fingerprint": "other"})):
            normal, attack = self.records()
            attack["round_results"][0][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                compare_accuracy_results(normal, attack)


if __name__ == "__main__":
    unittest.main()
