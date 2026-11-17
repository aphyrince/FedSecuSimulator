import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from common.model import create_model, save_model_checkpoint, weights_hash
from common.training import TrainingSettings
from exp_manager.services.experiment_runner import ExperimentManager


class RunningProcess:
    def poll(self):
        return None


class ReplayProcessManager:
    server_process = RunningProcess()

    def server_url(self):
        return "http://server"

    def client_url(self, client_id):
        return f"http://client{client_id}"

    def manager_url(self):
        return "http://manager"

    def client_registry(self):
        return [{"client_id": item, "url": self.client_url(item)} for item in range(3)]


class ReplayTests(unittest.TestCase):
    def test_round_replay_restores_data_ranges_and_start_checkpoint(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manager = ExperimentManager(ReplayProcessManager(), root, client_count=3)
            config = manager.get_config()
            config["experiment_id"] = "source-run"
            model = create_model(seed=42, config=config["model_config"])
            model["experiment_id"] = "source-run"
            checkpoint = root / "source-run" / "checkpoints" / "round_0001_start.pt"
            save_model_checkpoint(model, checkpoint)
            split = {
                "seed": 17,
                "split_id": "split-a",
                "source_dir": temporary,
                "test_range": {
                    "source_dir": temporary,
                    "ranges": [{"file": "test", "start": 0, "end": 10}],
                },
                "client_ranges": [
                    {"client_id": client_id, "ranges": [{"file": "train", "start": client_id, "end": client_id + 1}]}
                    for client_id in range(3)
                ],
                "file_sha256": {},
            }
            source = root / "source-run"
            source.mkdir(exist_ok=True)
            (source / "experiment_info.json").write_text(
                json.dumps({"experiment_type": "federated", "configuration": config, "data_split": split}),
                encoding="utf-8",
            )
            (source / "rounds").mkdir()
            round_config = {
                **TrainingSettings().to_dict(),
                "experiment_id": "source-run",
                "round": 1,
                "seed": 42,
                "nums_clients": 3,
                "selected_clients": [0, 1, 2],
                "malicious_clients": [],
                "attack_enabled": False,
                "defense_enabled": False,
                "defense": "none",
                "participation_rate": 1.0,
            }
            (source / "rounds" / "round_001.json").write_text(
                json.dumps(
                    {
                        "start_checkpoint": str(checkpoint),
                        "initial_model_hash": model["initial_model_hash"],
                        "start_model_hash": weights_hash(model["weights"]),
                        "round_config": round_config,
                    }
                ),
                encoding="utf-8",
            )
            posts = []
            with patch.object(
                manager.rest_client,
                "post_json",
                side_effect=lambda url, payload, **_kwargs: posts.append((url, payload)) or {},
            ), patch.object(
                manager, "execute_single_round", return_value={"round": 1}
            ) as execute:
                result = manager.replay_round("source-run", 1)

            self.assertEqual(result, {"round": 1})
            replay_payload = execute.call_args.args[1]
            replay_id = replay_payload["experiment_id"]
            self.assertNotEqual(replay_id, "source-run")
            self.assertEqual(replay_payload["initial_global_model"]["weights"], model["weights"])
            self.assertEqual(posts[0][1], split["test_range"])
            self.assertEqual(len(posts), 4)
            replay_info = json.loads(
                (root / replay_id / "experiment_info.json").read_text(encoding="utf-8")
            )
            self.assertEqual(replay_info["data_split"], split)
            self.assertEqual(replay_info["replay_of"], {"experiment_id": "source-run", "round": 1})


if __name__ == "__main__":
    unittest.main()
