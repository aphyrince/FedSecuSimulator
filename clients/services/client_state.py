"""클라이언트 프로세스의 실험 상태를 관리한다."""

from typing import Any
import threading
from pathlib import Path
import torch
from common.model import decode_weights, encode_weights, finite_number


class ClientState:
    """Flask route에서 사용하는 클라이언트 상태 모음."""

    def __init__(self, results_dir: Path | None = None) -> None:
        self.results_dir = results_dir or Path(__file__).resolve().parents[2] / "results"
        self.training_lock = threading.Lock()
        self.experiment_config: dict[str, Any] = {}
        self.dataset_range: dict[str, Any] = {}
        self.available = True
        self.global_model: dict[str, Any] = {}
        self.global_models: list[dict[str, Any]] = []
        self.local_training_results: list[dict[str, Any]] = []
        self.latest_local_training: dict[str, Any] = {}
        self.training_progress: dict[str, Any] = {
            "status": "idle",
            "round": None,
            "epoch": 0,
            "epochs": 0,
            "processed_samples": 0,
            "total_samples": 0,
            "remaining_samples": 0,
            "error": None,
        }
        self.extracted_updates: dict[int, dict[str, Any]] = {}
        self.submitted_updates: dict[int, dict[str, Any]] = {}
        self.expired_rounds: set[int] = set()
        self.malicious_peers: dict[int, list[dict[str, Any]]] = {}
        self.attack_scale_by_round: dict[int, float] = {}
        self.received_malicious_updates: dict[int, dict[int, dict[str, Any]]] = {}
        self.dlg_gradient: dict[str, Any] = {}
        self.dlg_round = 0

    def set_malicious_peers(
        self, round_id: int, peers: list[dict[str, Any]], attack_scale: float = -5.0
    ) -> list[dict[str, Any]]:
        """해당 라운드의 다른 악성 클라이언트 주소를 저장한다."""
        self.attack_scale_by_round[round_id] = finite_number(
            attack_scale, "attack_scale"
        )
        self.malicious_peers[round_id] = peers
        return peers

    def receive_malicious_update(
        self, round_id: int, sender_id: int, update: dict[str, Any]
    ) -> dict[str, Any]:
        """다른 악성 클라이언트로부터 받은 업데이트 추출값을 저장한다."""
        self.received_malicious_updates.setdefault(round_id, {})[sender_id] = update
        return update

    def set_experiment_config(self, config: dict[str, Any]) -> dict[str, Any]:
        self.experiment_config = config
        return self.experiment_config

    def set_dataset_range(self, dataset_range: dict[str, Any]) -> dict[str, Any]:
        self.dataset_range = dataset_range
        return self.dataset_range

    def begin_training(
        self,
        round_id: int,
        total_samples: int,
        epochs: int,
        *,
        malicious: bool,
        delayed: bool,
    ) -> None:
        self.training_progress = {
            "status": "training",
            "round": round_id,
            "epoch": 0,
            "epochs": epochs,
            "processed_samples": 0,
            "total_samples": total_samples,
            "remaining_samples": total_samples,
            "error": None,
            "malicious": malicious,
            "delayed": delayed,
        }

    def update_training_progress(
        self, epoch: int, epochs: int, processed_samples: int, total_samples: int
    ) -> None:
        self.training_progress.update(
            status="training",
            epoch=epoch,
            epochs=epochs,
            processed_samples=processed_samples,
            total_samples=total_samples,
            remaining_samples=max(0, total_samples - processed_samples),
        )

    def receive_global_model(self, model: dict[str, Any]) -> dict[str, Any]:
        if self.global_model.get("experiment_id") != model.get("experiment_id") or model.get("round") == 1:
            self.local_training_results.clear()
            self.extracted_updates.clear()
            self.received_malicious_updates.clear()
            self.expired_rounds.clear()
        self.global_model = model
        self.global_models.append({key: value for key, value in model.items() if key not in ("weights", "training_config")})
        return self.global_model

    def save_training(self, training_record: dict[str, Any]) -> None:
        round_id = int(training_record["round"])
        self.latest_local_training = training_record
        self.training_progress.update(
            status="trained",
            epoch=len(training_record.get("history", [])),
            processed_samples=training_record.get("num_samples", 0),
            total_samples=training_record.get("num_samples", 0),
            remaining_samples=0,
        )
        summary = {key: value for key, value in training_record.items() if key not in ("local_model", "model_delta", "extracted_update")}
        self.local_training_results.append(summary)
        extracted_update = training_record.get("extracted_update")
        if isinstance(extracted_update, dict):
            root = self.results_dir
            experiment_id = str(training_record["local_model"].get("experiment_id", "experiment"))
            path = root / experiment_id / "clients" / f"client_{training_record['client_id']}" / f"round_{round_id:04d}.pt"
            if not path.resolve().is_relative_to(root.resolve()):
                raise ValueError("Invalid experiment checkpoint path")
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"weights": decode_weights(training_record["local_model"]["weights"]),
                        "delta": decode_weights(extracted_update["values"]), "metadata": summary}, path)
            self.extracted_updates[round_id] = {"checkpoint": str(path), "update_norm": extracted_update["update_norm"]}

    def get_extracted_update(self, round_id: int):
        item = self.extracted_updates.get(round_id)
        if item is None:
            return None
        saved = torch.load(item["checkpoint"], map_location="cpu", weights_only=True)
        return {"values": encode_weights(saved["delta"]), "update_norm": item["update_norm"]}

    def save_submission(self, round_id: int, submission: dict[str, Any]) -> None:
        self.submitted_updates[round_id] = submission
        if self.training_progress.get("round") == round_id:
            self.training_progress["status"] = "submitted"
