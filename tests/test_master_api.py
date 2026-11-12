"""고정 업데이트를 반환하는 대체 함수로 프로세스 내부 Flask 경로를 검증한다. 실제 CNN은 학습하지 않는다."""

from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch
from urllib.parse import urlsplit

import torch

from clients.app import create_app as create_client_app
from clients.services.client_state import ClientState
from clients.services.training_service import build_round_update
from common.model import create_model, decode_weights, encode_weights, weights_hash, weights_norm, SmallCNN
from exp_manager.services.experiment_runner import ExperimentManager
from exp_manager.services.process_manager import ProcessManager
from server.app import create_app as create_server_app
from test_master_algorithms import make_update, vector_model


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        # 대체 함수 연결이 잘못되었으면 실제 학습이나 추론이 실행되기 전에 중단한다.
        for target in ("torch.Tensor.backward", "torch.optim.SGD", "common.model.SmallCNN.forward"):
            guard = patch(target, side_effect=AssertionError("Actual training/inference is forbidden in these tests"))
            guard.start()
            self.addCleanup(guard.stop)

    def test_no_completed_training_cannot_create_a_fake_update(self):
        state = ClientState(self.root)
        state.global_model = vector_model()
        with self.assertRaises(ValueError):
            build_round_update(state, 0, {"round": 1})

    def test_client_status_reports_training_progress_and_remaining_data(self):
        app = create_client_app(0, 5101)
        state = app.extensions["client_state"]
        state.set_dataset_range(
            {"ranges": [{"file": "train.bin", "start": 0, "end": 12}]}
        )
        state.global_model = {
            "round": 3,
            "training_config": {"max_train_samples": 8},
            "round_context": {"selected": True, "malicious": True, "delayed": False},
        }
        state.begin_training(3, 8, 2, malicious=True, delayed=False)
        state.update_training_progress(1, 2, 5, 8)

        status = app.test_client().get("/api/status").get_json()
        self.assertEqual(status["dataset_samples"], 12)
        self.assertEqual(status["trainable_samples"], 8)
        self.assertEqual(status["training"]["processed_samples"], 5)
        self.assertEqual(status["training"]["remaining_samples"], 3)
        self.assertTrue(status["malicious"])

    def test_server_status_reports_current_round_participation_and_aggregation(self):
        app = create_server_app("127.0.0.1", 5000)
        app.extensions["clients"] = {
            client_id: {"client_id": client_id, "url": f"http://client{client_id}"}
            for client_id in range(2)
        }
        app.extensions["round_progress"] = {
            "round": 2,
            "phase": "training",
            "message": "training",
        }
        app.extensions["round_config"] = {
            "participation_rate": 0.5,
            "attack_enabled": True,
            "malicious_clients": [1],
            "delay_clients": [0],
            "defense_enabled": True,
            "defense": "norm_clipping",
        }
        app.extensions["available_clients"][2] = [0, 1]
        app.extensions["availability_responses"][2] = [
            {"client_id": 0, "num_samples": 12},
            {"client_id": 1, "num_samples": 9},
        ]
        app.extensions["selected_clients"][2] = [0, 1]
        app.extensions["update_submission_status"][2] = {
            0: {"status": "waiting"},
            1: {"status": "received"},
        }
        app.extensions["round_updates"][2] = {1: {"num_samples": 9}}
        app.extensions.setdefault("expected_sample_counts", {})[2] = {0: 12, 1: 9}

        status = app.test_client().get("/api/status").get_json()
        self.assertEqual(status["round_phase"], "training")
        self.assertEqual(status["updates_received"], 1)
        self.assertTrue(status["client_activity"][0]["delayed"])
        self.assertTrue(status["client_activity"][1]["malicious"])
        self.assertEqual(status["client_activity"][1]["submitted_samples"], 9)
        self.assertEqual(status["client_activity"][1]["expected_samples"], 9)

    def test_process_manager_combines_server_and_client_live_status(self):
        class RunningProcess:
            def poll(self):
                return None

        manager = ProcessManager("127.0.0.1", 2, 5000, 5101)
        manager.server_process = RunningProcess()
        manager.client_processes = {0: RunningProcess(), 1: RunningProcess()}
        status_timeouts = {}

        def get_live_status(url, timeout):
            status_timeouts[url] = timeout
            if ":5000/" in url:
                return {
                    "current_round": 2,
                    "client_activity": [{"client_id": 1, "selected": True}],
                }
            client_id = int(url.split(":")[-1].split("/")[0]) - 5101
            return {"client_id": client_id, "training": {"status": "training"}}

        with patch.object(manager.rest_client, "get_json", side_effect=get_live_status):
            status = manager.status()

        self.assertTrue(status["server"]["reachable"])
        self.assertEqual(status["server"]["details"]["current_round"], 2)
        self.assertEqual(status["clients"][1]["details"]["training"]["status"], "training")
        self.assertTrue(status["clients"][1]["round_activity"]["selected"])
        self.assertEqual(status_timeouts["http://127.0.0.1:5000/api/status"], 2.0)
        self.assertEqual(status_timeouts["http://127.0.0.1:5101/api/status"], 0.3)

    def test_full_normal_and_attack_rounds_through_flask_with_stubs(self):
        starts = []
        submission_before_all_training_finished = []
        outcomes = {}
        for attack in (False, True):
            server = create_server_app("127.0.0.1", 5000)
            server.config["RESULTS_DIR"] = str(self.root)
            api = server.test_client()
            clients = {}
            for client_id, count in ((0, 2), (1, 6)):
                app = create_client_app(client_id, 5101 + client_id)
                app.config["SERVER_URL"] = "http://server"
                app.extensions["client_state"].results_dir = self.root
                app.extensions["client_state"].set_dataset_range({"source_dir": "unused", "ranges": [{"file": "fake", "start": 0, "end": count}]})
                clients[f"client{client_id}"] = app.test_client()
            server.extensions["clients"] = {i: {"client_id": i, "url": f"http://client{i}"} for i in range(2)}
            server.extensions["dataset_range"] = {"source_dir": "unused", "ranges": [{"file": "fake", "start": 0, "end": 10}]}
            config = {"experiment_id": "attack" if attack else "normal", "round": 1, "seed": 42,
                      "model_config": {"channels": [2, 3], "hidden_size": 4},
                      "nums_clients": 2, "participation_rate": 1.0, "delay_clients": [],
                      "malicious_clients": [0, 1], "attack_enabled": attack, "attack_scale": -5.0,
                      "defense_enabled": False, "defense": "none", "round_timeout": 60.0,
                      "local_epochs": 3, "batch_size": 7, "learning_rate": 0.02}
            self.assertEqual(api.post("/api/round/config", json=config).status_code, 200)
            initial = decode_weights(server.extensions["global_model"]["weights"])
            training_state = {"completed": 0}
            training_lock = threading.Lock()
            all_training_finished = threading.Event()

            def fake_train(global_model, dataset_range, training_config, round_id, client_id, **kwargs):
                self.assertEqual((training_config["local_epochs"], training_config["batch_size"], training_config["learning_rate"]), (3, 7, 0.02))
                starts.append((attack, client_id, weights_hash(global_model["weights"])))
                base = decode_weights(global_model["weights"])
                delta = {key: torch.full_like(value, 1.0 if client_id == 0 else 3.0) for key, value in base.items()}
                update = make_update(global_model, client_id, delta, samples=2 if client_id == 0 else 6)
                result = {"local_model": update["local_model"], "model_delta": encode_weights(delta),
                          "num_samples": update["num_samples"], "training_loss": 1.0,
                          "base_model_hash": weights_hash(base), "base_model_version": global_model["model_version"],
                          "dataset_fingerprint": f"train-{client_id}", "training_config": training_config}
                with training_lock:
                    training_state["completed"] += 1
                    if training_state["completed"] == 2:
                        all_training_finished.set()
                return result

            def dispatch(url, payload, timeout=3.0):
                parsed = urlsplit(url)
                if parsed.path == "/api/model/global":
                    global_delivery_barrier.wait(timeout=5)
                target = server.test_client() if parsed.netloc == "server" else clients[parsed.netloc]
                response = target.post(parsed.path, json=payload)
                if response.status_code >= 400:
                    raise ValueError(f"{parsed.path}: {response.get_json()}")
                if parsed.netloc == "server" and parsed.path.endswith("/updates"):
                    submission_before_all_training_finished.append(
                        not all_training_finished.is_set()
                    )
                return response.get_json()

            global_delivery_barrier = threading.Barrier(2)
            evaluation = {"accuracy": 0.25, "loss": 1.0, "num_test_samples": 10, "correct_predictions": 2,
                          "metrics": {}, "dataset_fingerprint": "same-test"}
            with patch("clients.services.training_service.train_local_model", side_effect=fake_train), \
                    patch("clients.routes.experiment.post_json", side_effect=dispatch), \
                    patch("server.routes.aggregation._post_json", side_effect=dispatch), \
                    patch("server.routes.aggregation.evaluate_global_model", return_value=evaluation):
                response = api.post("/api/rounds/1/start", json={**config, "server_url": "http://server"})
            self.assertEqual(response.status_code, 200, response.get_json())
            result = response.get_json()
            self.assertEqual(result["accepted_clients"], [0, 1])
            self.assertEqual(result["attack_applied"], attack)
            expected_delta = -10.0 if attack else 2.5
            final = decode_weights(server.extensions["global_model"]["weights"])
            for key in initial:
                torch.testing.assert_close(final[key], initial[key] + expected_delta, atol=2e-6, rtol=1e-5)
            self.assertTrue(Path(result["global_model"]["checkpoint"]).is_file())
            self.assertNotIn("weights", result["global_model"])
            self.assertNotIn("values", result["client_updates"][0]["update"])
            self.assertEqual(result["accuracy"], 0.25)
            self.assertEqual(api.post("/api/rounds/1/start", json={**config, "server_url": "http://server"}).status_code, 409)
            outcomes[attack] = result
        self.assertEqual(starts[0][2], starts[1][2])
        self.assertEqual(starts[0][2], starts[2][2])
        self.assertEqual(outcomes[False]["initial_model_hash"], outcomes[True]["initial_model_hash"])
        self.assertEqual(submission_before_all_training_finished, [False] * 4)

    def test_failed_evaluation_returns_failure_without_fake_accuracy_or_model_commit(self):
        app = create_server_app("127.0.0.1", 5000)
        app.config["RESULTS_DIR"] = str(self.root)
        model = vector_model()
        app.extensions["global_model"] = model
        update = make_update(model, 0, {"w": [1.0, 2.0]})
        with patch("server.routes.aggregation.evaluate_global_model", side_effect=ValueError("missing test dataset")):
            response = app.test_client().post("/api/rounds/1/execute", json={"experiment_id": "test", "selected_clients": [0], "updates": [update]})
        self.assertEqual(response.status_code, 400)
        self.assertNotIn("accuracy", response.get_json())
        self.assertEqual(app.extensions["global_model"], model)
        self.assertFalse(app.extensions["round_results"])

    def test_update_endpoint_rejects_duplicate_late_and_invalid_updates(self):
        app = create_server_app("127.0.0.1", 5000)
        model = vector_model()
        app.extensions["global_model"] = model
        app.extensions["selected_clients"][1] = [0]
        update = make_update(model, 0, {"w": [1.0, 2.0]})
        api = app.test_client()
        self.assertEqual(api.post("/api/rounds/1/updates", json={**update, "num_samples": 0}).status_code, 400)
        self.assertEqual(api.post("/api/rounds/1/updates", json=update).status_code, 200)
        self.assertEqual(api.post("/api/rounds/1/updates", json=update).status_code, 409)
        app.extensions["update_deadlines"] = {1: (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat()}
        self.assertEqual(api.post("/api/rounds/1/updates", json=update).status_code, 408)

    def test_concurrent_submissions_match_server_and_manager_status(self):
        client_count = 10
        app = create_server_app("127.0.0.1", 5000)
        model = vector_model()
        app.extensions["global_model"] = model
        app.extensions["selected_clients"][1] = list(range(client_count))
        app.extensions["available_clients"][1] = list(range(client_count))
        app.extensions["round_progress"].update(
            round=1,
            phase="training",
            config={"participation_rate": 1.0, "attack_enabled": False},
        )
        app.extensions["clients"] = {
            client_id: {
                "client_id": client_id,
                "url": f"http://127.0.0.1:{5101 + client_id}",
            }
            for client_id in range(client_count)
        }
        app.extensions["update_submission_status"][1] = {
            client_id: {"status": "waiting"} for client_id in range(client_count)
        }
        updates = [
            make_update(model, client_id, {"w": [1.0, 2.0]}, samples=2)
            for client_id in range(client_count)
        ]

        def submit(update):
            with app.test_client() as client:
                response = client.post("/api/rounds/1/updates", json=update)
                return response.status_code, response.get_json()

        with ThreadPoolExecutor(max_workers=client_count) as executor:
            responses = list(executor.map(submit, updates))

        self.assertTrue(all(status == 200 for status, _ in responses))
        stored_updates = app.extensions["round_updates"][1]
        self.assertEqual(set(stored_updates), set(range(client_count)))

        api = app.test_client()
        server_status = api.get("/api/status").get_json()
        submission_status = api.get("/api/rounds/1/updates/status").get_json()
        self.assertEqual(server_status["updates_received"], len(stored_updates))
        self.assertEqual(
            submission_status["submitted_clients"], sorted(stored_updates)
        )
        self.assertTrue(
            all(
                item["update_status"] == "received"
                for item in server_status["client_activity"]
            )
        )

        class RunningProcess:
            def poll(self):
                return None

        manager = ProcessManager("127.0.0.1", 0, 5000, 5101)
        manager.server_process = RunningProcess()
        with patch.object(
            manager.rest_client,
            "get_json",
            side_effect=lambda _url, timeout: api.get("/api/status").get_json(),
        ):
            manager_status = manager.status()
        self.assertEqual(
            manager_status["server"]["details"]["updates_received"],
            len(stored_updates),
        )

    def test_manager_propagates_settings_rolls_back_invalid_values_and_respects_candidates(self):
        class Processes:
            def client_registry(self):
                return [{"client_id": i, "ip": "127.0.0.1", "port": 5101 + i, "url": f"http://client{i}"} for i in range(10)]

        manager = ExperimentManager(Processes(), self.root, 10)
        manager.update_config({
            "learning_rate": 0.02,
            "local_epochs": 4,
            "malicious_clients": [2, 7],
            "attack_enabled": True,
            "attack_scale": -2.5,
        })
        config = manager.get_config()
        round_config = manager._round_config(1, config)
        self.assertEqual((round_config["learning_rate"], round_config["local_epochs"]), (0.02, 4))
        with patch.object(manager.rest_client, "post_json", return_value={}):
            self.assertEqual(manager.prepare_malicious_clients(1, round_config, config), [2, 7])
            self.assertEqual(manager.rest_client.post_json.call_count, 2)
            self.assertTrue(
                all(
                    call.args[1]["attack_scale"] == -2.5
                    for call in manager.rest_client.post_json.call_args_list
                )
            )
        with self.assertRaises(ValueError):
            manager.update_config({"learning_rate": -1})
        with self.assertRaises(ValueError):
            manager.update_config({"attack_scale": float("nan")})
        self.assertEqual(manager.get_config()["learning_rate"], 0.02)
        self.assertEqual(manager.get_config()["attack_scale"], -2.5)

    def test_client_uses_scale_received_from_manager_for_collusion(self):
        app = create_client_app(0, 5101)
        api = app.test_client()
        configured = api.post(
            "/api/round/malicious-peers",
            json={"round": 1, "peers": [], "attack_scale": -2.5},
        )
        self.assertEqual(configured.status_code, 200)

        own_update = {"client_id": 0, "round": 1}
        with patch(
            "clients.routes.experiment.collude_updates",
            side_effect=lambda model, own, peers, scale, minimum: {"scale": scale},
        ) as collude:
            response = api.post(
                "/api/round/collusion",
                json={
                    "round": 1,
                    "own_update": own_update,
                    "peer_updates": [{"client_id": 1}],
                    "attack_scale": -99,
                },
            )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["scale"], -2.5)
        self.assertEqual(collude.call_args.args[3], -2.5)

    def test_callback_does_not_advance_until_server_request_returns(self):
        manager = ExperimentManager(object(), self.root, 10)
        manager.config.update(experiment_id="callback_test", num_rounds=2)
        manager.state["status"] = "running"
        result = {"experiment_id": "callback_test", "round": 1, "accuracy": 0.5, "loss": 1.0}
        with patch("exp_manager.services.experiment_runner.threading.Thread") as thread:
            manager.record_round_result(result)
            thread.assert_not_called()
            manager.record_round_result(result, advance=True)
            self.assertEqual(thread.call_count, 1)
            manager.record_round_result(result, advance=True)
            self.assertEqual(thread.call_count, 1)

    def test_stopping_experiment_cannot_reset_or_change_configuration(self):
        manager = ExperimentManager(object(), self.root, 10)
        manager.state["status"] = "stopping"
        with self.assertRaises(RuntimeError):
            manager.reset()
        with self.assertRaises(ValueError):
            manager.update_config({"learning_rate": 0.5})
        with self.assertRaises(RuntimeError):
            manager.start()

    def test_model_configuration_cannot_reset_a_running_round(self):
        app = create_server_app("127.0.0.1", 5000)
        import threading
        lock = threading.Lock()
        app.extensions["round_lock"] = lock
        lock.acquire()
        try:
            self.assertEqual(app.test_client().post("/api/round/config", json={"round": 1}).status_code, 409)
        finally:
            lock.release()

    def test_actual_evaluation_counts_last_batch_correctly_with_fixed_logits(self):
        from torch.utils.data import TensorDataset
        from server.services.model_evaluation import evaluate_global_model
        dataset = TensorDataset(torch.arange(3), torch.tensor([0, 1, 1]))
        dataset.fingerprint = "fixed-data"
        logits = torch.tensor([[4.0, 0.0], [0.0, 4.0], [4.0, 0.0]])

        class FixedModel:
            def to(self, device):
                return self

            def eval(self):
                return self

            def __call__(self, indices):
                return logits[indices]

        with patch("server.services.model_evaluation.CifarRangeDataset", return_value=dataset), \
                patch("server.services.model_evaluation.build_model", return_value=FixedModel()), \
                patch("fedsec.training.setup_runtime", return_value=torch.device("cpu")):
            result = evaluate_global_model({}, {}, {"batch_size": 2}, 1)
        self.assertEqual((result["num_test_samples"], result["correct_predictions"]), (3, 2))
        self.assertAlmostEqual(result["accuracy"], 2 / 3)
        expected = torch.nn.functional.cross_entropy(logits, torch.tensor([0, 1, 1])).item()
        self.assertAlmostEqual(result["loss"], expected, places=6)


if __name__ == "__main__":
    unittest.main()
