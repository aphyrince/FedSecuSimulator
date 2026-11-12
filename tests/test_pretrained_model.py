import json
import tempfile
from pathlib import Path

from common.model import create_model
from common.training import TrainingSettings
from exp_manager.services.experiment_runner import ExperimentManager
from exp_manager.services.pretrained_model import parse_pretrained_global_model
from server.app import create_app


class UnusedProcessManager:
    def server_url(self):
        return "http://127.0.0.1:5000"


class FinalModelRestClient:
    def __init__(self, model):
        self.model = model

    def get_json(self, url, timeout):
        assert url.endswith("/api/model/global")
        assert timeout == 60.0
        return self.model


def small_model():
    return create_model(seed=7, config={"channels": [1, 1], "hidden_size": 2})


def test_parse_json_and_flat_checkpoint_formats():
    model = small_model()
    parsed_json = parse_pretrained_global_model(
        json.dumps(model), "json", model["model_config"]
    )
    assert parsed_json["weights"] == model["weights"]
    assert parsed_json["model_config"] == model["model_config"]

    flat_values = [
        value
        for tensor in model["weights"].values()
        for value in _flatten(tensor)
    ]
    parsed_csv = parse_pretrained_global_model(
        ",".join(map(str, flat_values)), "csv", model["model_config"]
    )
    parsed_text = parse_pretrained_global_model(
        " ".join(map(str, flat_values)), "text", model["model_config"]
    )
    assert parsed_csv["weights"] == parsed_text["weights"]
    assert parsed_csv["weights"] == model["weights"]


def _flatten(values):
    if isinstance(values, list):
        for value in values:
            yield from _flatten(value)
    else:
        yield values


def test_pretrained_dlg_config_skips_training_rounds():
    with tempfile.TemporaryDirectory() as temporary_directory:
        manager = ExperimentManager(
            UnusedProcessManager(), Path(temporary_directory), client_count=10
        )
        model = small_model()
        manager.update_config(
            {
                "experiment_type": "dlg",
                "use_pretrained_global_model": True,
                "pretrained_model_format": "json",
                "pretrained_model_data": json.dumps(model),
            }
        )

        assert manager.total_round_count() == 1
        assert manager._round_config(1, manager.get_config())["round_type"] == "dlg"
        assert manager._round_config(2, manager.get_config())["round_type"] == "federated"
        assert manager.get_public_config()["pretrained_global_model_loaded"] is True
        assert "pretrained_global_model" not in manager.get_public_config()
        assert "pretrained_global_model" not in manager.status()["config"]


def test_server_accepts_pretrained_checkpoint_as_initial_global_model():
    model = small_model()
    with tempfile.TemporaryDirectory() as temporary_directory:
        app = create_app("127.0.0.1", 5000)
        app.config["RESULTS_DIR"] = Path(temporary_directory)
        payload = {
            **TrainingSettings().to_dict(),
            "experiment_id": "pretrained-dlg-test",
            "round": 1,
            "seed": 11,
            "model_config": model["model_config"],
            "initial_global_model": model,
        }

        response = app.test_client().post("/api/round/config", json=payload)

        assert response.status_code == 200
        assert app.extensions["global_model"]["weights"] == model["weights"]
        assert app.extensions["global_model"]["experiment_id"] == "pretrained-dlg-test"
        assert "initial_global_model" not in app.extensions["round_config"]


def test_federated_completion_exports_final_global_model_for_dlg():
    with tempfile.TemporaryDirectory() as temporary_directory:
        model = {**small_model(), "experiment_id": "final-model-export"}
        manager = ExperimentManager(
            UnusedProcessManager(), Path(temporary_directory), client_count=10
        )
        manager.update_config(
            {"experiment_id": "final-model-export", "num_rounds": 1}
        )
        manager.rest_client = FinalModelRestClient(model)
        manager.state["status"] = "running"

        manager.record_round_result(
            {
                "experiment_id": "final-model-export",
                "round": 1,
                "accuracy": 0.75,
                "loss": 0.5,
            },
            advance=True,
        )

        export_path = (
            Path(temporary_directory)
            / "final-model-export"
            / "global_model_for_dlg.json"
        )
        exported = json.loads(export_path.read_text(encoding="utf-8"))
        assert exported["weights"] == model["weights"]
        imported = parse_pretrained_global_model(
            export_path.read_text(encoding="utf-8"),
            "json",
            model["model_config"],
        )
        assert imported["weights"] == model["weights"]
        assert manager.state["status"] == "completed"
        assert manager.state["global_model_export_path"] == str(export_path)
