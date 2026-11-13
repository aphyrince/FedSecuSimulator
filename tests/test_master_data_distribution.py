import json
import pickle
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

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
                    pickle.dump({b"data": [[0] * 3072 for _ in range(count)]}, handle)

            manager = DataDistributionManager(process_manager=None, client_count=2)
            manager.loader.resolve_files = lambda _source, _names, kind: [
                train_path if kind == "train" else test_path
            ]
            manager._send_test_range = lambda _payload: None
            manager._send_client_ranges = lambda source_dir, ranges: [
                {"client_id": index, "ranges": item}
                for index, item in enumerate(ranges)
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
            self.assertEqual(set(manifest["file_sha256"]), {"train", "test"})
            self.assertTrue(status["split_id"])


if __name__ == "__main__":
    unittest.main()
