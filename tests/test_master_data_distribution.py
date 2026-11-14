import json
import pickle
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from fedsec.data import CifarRangeDataset
from exp_manager.services.data_distribution import DataDistributionManager


class DataDistributionTests(unittest.TestCase):
    def test_seeded_split_manifest_persists_ranges_and_dataset_hashes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "cifar"
            source.mkdir()
            train_path = source / "train"
            test_path = source / "test"
            for path, count in ((train_path, 6), (test_path, 3)):
                with path.open("wb") as handle:
                    pickle.dump(
                        {
                            b"data": [[0] * 3072 for _ in range(count)],
                            b"labels": [index % 10 for index in range(count)],
                        },
                        handle,
                    )

            manager = DataDistributionManager(process_manager=None, client_count=2)
            manager.loader.resolve_files = lambda _source, _names, kind: [
                train_path if kind == "train" else test_path
            ]
            manager._send_test_range = lambda _payload: None
            manager._send_client_ranges = lambda source_dir, assignments: [
                {"client_id": index, **item}
                for index, item in enumerate(assignments)
            ]
            config = {
                "source_path": str(source),
                "source_dir": str(source),
                "train_files": [],
                "test_files": [],
            }
            with patch(
                "exp_manager.services.data_distribution.read_data_config",
                return_value=config,
            ), patch(
                "exp_manager.services.data_distribution.PROJECT_ROOT_PATH", root
            ):
                status = manager.distribute(seed=17)

            manifest_path = Path(status["split_manifest_path"])
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(status["seed"], 17)
            self.assertEqual(manifest["seed"], 17)
            self.assertEqual(manifest["client_ranges"], status["client_ranges"])
            self.assertEqual(manifest["test_range"], status["test_range"])
            self.assertEqual(manifest["distribution"], "iid")
            self.assertEqual(
                manifest["client_class_counts"], status["client_class_counts"]
            )
            self.assertEqual(set(manifest["file_sha256"]), {"train", "test"})
            self.assertTrue(status["split_id"])

    def test_non_iid_distribution_assigns_label_skewed_indices_and_records_counts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "cifar"
            source.mkdir()
            train_path = source / "train"
            test_path = source / "test"
            labels = [label for label in range(10) for _ in range(10)]
            for path, batch_labels in ((train_path, labels), (test_path, [0, 1, 2])):
                with path.open("wb") as handle:
                    pickle.dump(
                        {
                            b"data": [[0] * 3072 for _ in batch_labels],
                            b"labels": batch_labels,
                        },
                        handle,
                    )

            manager = DataDistributionManager(process_manager=None, client_count=10)
            manager.loader.resolve_files = lambda _source, _names, kind: [
                train_path if kind == "train" else test_path
            ]
            manager._send_test_range = lambda _payload: None
            manager._send_client_ranges = lambda _source, assignments: [
                {"client_id": index, **item}
                for index, item in enumerate(assignments)
            ]
            config = {
                "source_path": str(source),
                "source_dir": str(source),
                "train_files": [],
                "test_files": [],
            }
            with patch(
                "exp_manager.services.data_distribution.read_data_config",
                return_value=config,
            ), patch(
                "exp_manager.services.data_distribution.PROJECT_ROOT_PATH", root
            ):
                status = manager.distribute(seed=7, distribution="non_iid")

            manifest = json.loads(
                Path(status["split_manifest_path"]).read_text(encoding="utf-8")
            )
            assignments = manifest["client_ranges"]
            assert manifest["distribution"] == "non_iid"
            assert len(assignments) == 10
            for assignment, counts in zip(
                assignments, manifest["client_class_counts"]
            ):
                assert assignment["indices_by_file"]
                assigned_labels = [
                    labels[index]
                    for segment in assignment["indices_by_file"]
                    for index in segment["indices"]
                ]
                assert len(set(assigned_labels)) == 2
                assert counts["num_samples"] == len(assigned_labels)
                assert sum(counts["class_counts"].values()) == len(assigned_labels)
            all_indices = [
                index
                for assignment in assignments
                for segment in assignment["indices_by_file"]
                for index in segment["indices"]
            ]
            assert sorted(all_indices) == list(range(len(labels)))

    def test_client_dataset_loads_label_skew_indices(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "cifar"
            source.mkdir()
            labels = [8, 2, 8, 3]
            with (source / "train").open("wb") as handle:
                pickle.dump(
                    {
                        b"data": np.zeros((len(labels), 3072), dtype=np.uint8),
                        b"labels": labels,
                    },
                    handle,
                )
            dataset = CifarRangeDataset(
                {
                    "source_dir": str(source),
                    "indices_by_file": [
                        {"file": "train", "indices": [0, 2, 3]}
                    ],
                }
            )
            self.assertEqual(len(dataset), 3)
            self.assertEqual(
                [dataset[index][1] for index in range(len(dataset))], [8, 8, 3]
            )


if __name__ == "__main__":
    unittest.main()
