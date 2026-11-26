"""Run virtual clients serially against the central Flask server."""

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import threading
import time
from urllib.error import HTTPError

from clients.services.client_state import ClientState
from clients.services.local_training import count_dataset_samples
from clients.services.server_client import post_json
from clients.services.training_service import build_round_update, run_local_training
from common.model import decode_weights, encode_wire_weights, weights_hash
from common.training import TrainingSettings
from exp_manager.services.rest_client import RestClient


class SequentialClientRunner:
    def __init__(self, client_count: int, results_dir: Path, server_url: str):
        self.clients = {client_id: ClientState(results_dir) for client_id in range(client_count)}
        self.results_dir = results_dir
        self.server_url = server_url
        self.rest = RestClient()
        self.lock = threading.Lock()

    def configure_datasets(self, source_dir: str, assignments: list[dict]) -> None:
        if len(assignments) != len(self.clients):
            raise ValueError("순차 클라이언트 수와 데이터 배정 수가 다릅니다.")
        with self.lock:
            for client_id, assignment in enumerate(assignments):
                self.clients[client_id].set_dataset_range({"source_dir": source_dir, **assignment})

    def sample_counts(self) -> list[dict]:
        return [
            {"client_id": client_id, "num_samples": count_dataset_samples(state.dataset_range)}
            for client_id, state in self.clients.items()
        ]

    def client_status(self, client_id: int) -> dict:
        state = self.clients[client_id]
        samples = count_dataset_samples(state.dataset_range)
        training = dict(state.training_progress)
        return {
            "client_id": client_id,
            "process_id": os.getpid(),
            "execution": "sequential",
            "available": bool(samples) and not state.training_lock.locked(),
            "dataset_samples": samples,
            "trainable_samples": training.get("total_samples") or samples,
            "current_round": training.get("round"),
            "training": training,
        }

    def run_round(self, config: dict, started: dict, stop_event: threading.Event) -> None:
        round_id = config["round"]
        experiment_id = config["experiment_id"]
        selected = started["selected_clients"]
        schedule = config["sequential_schedule"]
        if (
            selected != schedule["selected_clients"] or started["expected_clients"] != selected
            or started.get("sequential_schedule") != schedule
        ):
            raise RuntimeError("중앙 서버의 참여 일정이 저장된 일정과 다릅니다.")
        with self.lock:
            started_at = datetime.now(timezone.utc).isoformat()
            started_clock = time.monotonic()
            model = self.rest.get_json(f"{self.server_url}/api/model/global", timeout=120.0)
            model["weights"] = decode_weights(model["weights"])
            model_hash = weights_hash(model["weights"])
            if (
                model.get("experiment_id") != experiment_id
                or model.get("model_version") != started["start_model_version"]
                or model_hash != started["start_model_hash"]
            ):
                raise RuntimeError("다운로드한 모델이 이 라운드의 시작 모델과 다릅니다.")
            base_model = {
                **model, "round": round_id, "update_deadline": started["deadline"],
                "training_config": {**TrainingSettings.from_dict(config).to_dict(), "seed": config["seed"]},
            }
            report = {
                "experiment_id": experiment_id, "seed": config["seed"], "round": round_id, "process_id": os.getpid(),
                "model_downloads": 1, "base_model_version": model["model_version"],
                "base_model_hash": model_hash, "selected_clients": selected,
                "unselected_clients": schedule["unselected_clients"], "schedule": schedule,
                "real_started_at": started_at,
            }
            entries = {}
            for client_id, state in self.clients.items():
                arrival = schedule["virtual_arrival_seconds"].get(str(client_id))
                chosen = client_id in selected
                entries[client_id] = {
                    "experiment_id": experiment_id, "seed": config["seed"], "round": round_id,
                    "client_id": client_id, "selected": chosen,
                    "training_status": "pending" if chosen else "not_selected",
                    "virtual_arrival_seconds": arrival, "virtual_elapsed_seconds": arrival,
                    "virtual_deadline_seconds": schedule["virtual_deadline_seconds"],
                }
                if not chosen:
                    state.training_progress = {"round": round_id, "status": "not_selected"}
            report["clients"] = list(entries.values())
            path = self._log_path(experiment_id, round_id)
            self._write_log(path, report)
            for client_id in selected:
                if stop_event.is_set():
                    entry = {"client_id": client_id, "training_status": "not_started"}
                    self._report_failure(round_id, client_id, "experiment_stopped", entry)
                else:
                    entry = self._train_and_submit(client_id, base_model)
                entries[client_id].update(entry)
                self._write_log(path, report)
            report.update(real_completed_at=datetime.now(timezone.utc).isoformat(),
                          real_elapsed_seconds=time.monotonic() - started_clock)
            self._write_log(path, report)
            self.rest.post_json(
                f"{self.server_url}/api/rounds/{round_id}/close",
                {"experiment_id": experiment_id}, timeout=10.0,
            )

    def _train_and_submit(self, client_id: int, base_model: dict) -> dict:
        state = self.clients[client_id]
        round_id = base_model["round"]
        entry = {"client_id": client_id}
        with state.training_lock:
            # The adapter constructs a fresh PyTorch model from this read-only snapshot.
            state.receive_global_model(base_model)
            try:
                trained = run_local_training(state, client_id, base_model, tensor_payload=True)
                entry.update({key: trained[key] for key in (
                    "training_started_at", "training_completed_at", "training_duration_seconds",
                    "base_model_version", "base_model_hash", "num_samples", "dataset_fingerprint", "training_accuracy",
                )})
                entry["training_status"] = "completed"
                update = build_round_update(state, client_id, {"round": round_id}, tensor_payload=True)
                entry.update(local_model_version=update["local_model"]["model_version"],
                             local_model_hash=weights_hash(update["local_model"]["weights"]),
                             update_norm=update["update"]["update_norm"])
                update["local_model"]["weights"] = encode_wire_weights(update["local_model"]["weights"])
                update["update"]["values"] = encode_wire_weights(update["update"]["values"])
                update["tensor_encoding"] = "base64_float32"
            except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
                state.fail_training(round_id, str(error))
                if entry.get("training_status") == "completed":
                    entry["submission"] = {"outcome": "preparation_failure", "error": str(error)}
                    self._report_failure(round_id, client_id, "update_preparation_failure", entry)
                else:
                    entry.update(training_status="failed", training_error=str(error))
                    self._report_failure(round_id, client_id, "real_training_failure", entry)
                return entry
            finally:
                # ClientState already saved the checkpoint and lightweight history.
                state.latest_local_training = {}
            entry["submission"] = self._submit_once(state, client_id, round_id, update)
        return entry

    def _submit_once(self, state: ClientState, client_id: int, round_id: int, update: dict) -> dict:
        url = f"{self.server_url}/api/rounds/{round_id}/updates"
        try:
            response = post_json(url, update, timeout=60.0, compact=True)
        except HTTPError as error:
            outcome = {"outcome": "http_error", "http_status": error.code, "error": str(error)}
        except (OSError, ValueError) as error:
            timed_out = isinstance(error, TimeoutError) or isinstance(getattr(error, "reason", None), TimeoutError)
            outcome = {"outcome": "response_timeout" if timed_out else "response_unavailable", "error": str(error)}
        else:
            state.save_submission(round_id, response)
            return {"outcome": "response_received", "response": response}

        # A lost response is ambiguous. Only query the server; this update is never posted again.
        state.training_progress.update(status="awaiting_server_confirmation")
        try:
            tracked = self.rest.get_json(f"{url}/status", timeout=10.0)["status"].get(str(client_id), {})
            outcome["server_status_after_response"] = tracked
        except (OSError, ValueError, KeyError) as error:
            outcome["status_query_error"] = str(error)
        return outcome

    def _report_failure(self, round_id: int, client_id: int, reason: str, entry: dict) -> None:
        try:
            self.rest.post_json(f"{self.server_url}/api/rounds/{round_id}/failures", {
                "round": round_id, "client_id": client_id, "reason": reason,
                "error": entry.get("training_error", ""),
            })
        except (OSError, ValueError) as error:
            entry["failure_report_error"] = str(error)

    def record_server_result(self, result: dict) -> tuple[Path, dict] | None:
        path = self._log_path(result["experiment_id"], result["round"])
        with self.lock:
            if not path.is_file():
                return None
            report = json.loads(path.read_text(encoding="utf-8"))
            activity = {row["client_id"]: row for row in result["client_activity"]}
            evaluations = {row["client_id"]: row for row in (result.get("evaluation") or {}).get("client_evaluations", [])}
            trained = []
            for entry in report["clients"]:
                client_id = entry["client_id"]
                if entry["training_status"] == "completed":
                    trained.append(client_id)
                row = activity[client_id]
                entry["server_submission"] = {key: row[key] for key in (
                    "status", "submission_status", "reason", "received_at", "accepted_at",
                )}
                entry["client_evaluation"] = evaluations.get(client_id)
                state = self.clients[client_id]
                if entry["selected"] and state.training_progress.get("round") == result["round"]:
                    state.training_progress.update(status=row["status"], error=row["reason"])
            summary = {
                "trained_clients": trained, "trained_count": len(trained),
                "selected_count": len(result["selected_clients"]),
                "unselected_clients": report["unselected_clients"],
                "unselected_count": len(report["unselected_clients"]),
                "accepted_clients": result["accepted_clients"], "accepted_count": len(result["accepted_clients"]),
                "excluded_clients": result["excluded_clients"], "excluded_count": len(result["excluded_clients"]),
            }
            report.update(summary)
            report["server_result"] = {key: result[key] for key in (
                "accepted_clients", "model_updated", "model_version", "start_model_hash", "fedavg_samples",
                "start_model_version", "end_model_version", "end_model_hash",
            )}
            self._write_log(path, report)
        return path, summary

    def _log_path(self, experiment_id: str, round_id: int) -> Path:
        return self.results_dir / experiment_id / "client_execution" / f"round_{round_id:03d}.json"

    @staticmethod
    def _write_log(path: Path, report: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_suffix(".tmp")
        temp.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        temp.replace(path)
