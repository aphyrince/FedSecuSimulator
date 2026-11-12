from datetime import datetime, timedelta, timezone
import json
import random
import threading
from copy import deepcopy
from pathlib import Path
from functools import wraps
from urllib.error import URLError
from urllib.request import Request, urlopen

from flask import Blueprint, current_app, jsonify, request

from server.services.fedavg import aggregate_coordinate_median, aggregate_fedavg
from server.services.model_evaluation import evaluate_global_model
from server.services.byzantine_defense import apply_norm_clipping
from common.model import decode_weights, weights_norm, save_model_checkpoint, positive_int
from common.updates import update_delta, validate_updates
from common.training import TrainingSettings


aggregation_bp = Blueprint("server_aggregation", __name__)


def one_round_at_a_time(function):
    @wraps(function)
    def wrapped(*args, **kwargs):
        lock = current_app.extensions.setdefault("round_lock", threading.Lock())
        if not lock.acquire(blocking=False):
            return jsonify(error="A round is already running"), 409
        try:
            return function(*args, **kwargs)
        finally:
            timer = current_app.extensions.pop("round_timer", None)
            if timer is not None:
                timer.cancel()
            lock.release()
    return wrapped


@aggregation_bp.post("/api/rounds/<int:round_id>/updates")
def receive_client_update(round_id: int):
    """클라이언트가 제출한 라운드 업데이트를 검증해 라운드별로 저장한다."""
    payload = request.get_json(silent=True) or {}
    if round_id in current_app.extensions.get("completed_rounds", set()):
        return jsonify(error="Round is already completed"), 409
    deadline = current_app.extensions.get("update_deadlines", {}).get(round_id)
    if deadline and datetime.now(timezone.utc) >= datetime.fromisoformat(deadline):
        return jsonify(error="Submission deadline expired"), 408
    try:
        positive_int(payload.get("round"), "round")
        positive_int(payload.get("client_id"), "client_id", 0)
    except ValueError as error:
        return jsonify(error=str(error)), 400
    if payload.get("round", round_id) != round_id:
        return jsonify(error="Round number mismatch"), 400
    client_id = payload.get("client_id")
    if client_id is None or not payload.get("valid", False):
        return jsonify(error="Invalid client update"), 400
    selected = current_app.extensions.get("selected_clients", {}).get(round_id, [])
    if int(client_id) not in selected:
        return jsonify(error="Client was not selected for this round"), 403
    num_samples = payload.get("num_samples")
    if not isinstance(num_samples, (int, float)) or num_samples < 0:
        return jsonify(error="num_samples must be a non-negative number"), 400
    if "update" not in payload or not isinstance(payload["update"], dict):
        return jsonify(error="Invalid update payload"), 400
    try:
        update_delta(current_app.extensions["global_model"], payload)
        expected = current_app.extensions.get("expected_sample_counts", {}).get(round_id, {}).get(int(client_id))
        if expected is not None and num_samples != expected:
            raise ValueError("Submitted sample count does not match assigned data")
    except (ValueError, KeyError, TypeError) as error:
        return jsonify(error=str(error)), 400

    updates = current_app.extensions.setdefault("round_updates", {})
    round_updates = updates.setdefault(round_id, {})
    if int(client_id) in round_updates:
        return jsonify(error="Duplicate client update"), 409
    round_updates[int(client_id)] = payload
    statuses = current_app.extensions.setdefault("update_submission_status", {})
    round_status = statuses.setdefault(round_id, {})
    round_status[int(client_id)] = {
        "status": "received",
        "num_samples": num_samples,
        "received_at": datetime.now(timezone.utc).isoformat(),
    }
    return jsonify(
        status="received",
        round=round_id,
        client_id=int(client_id),
        count=len(round_updates),
    )


@aggregation_bp.get("/api/rounds/<int:round_id>/updates")
def get_client_updates(round_id: int):
    """라운드별로 수신된 클라이언트 업데이트 목록을 반환한다."""
    updates = current_app.extensions.get("round_updates", {}).get(round_id, {})
    return jsonify(
        round=round_id,
        count=len(updates),
        updates=[updates[client_id] for client_id in sorted(updates)],
    )


@aggregation_bp.get("/api/rounds/<int:round_id>/updates/status")
def get_update_submission_status(round_id: int):
    """선정 클라이언트의 제출, 미제출, 기한 초과 상태를 반환한다."""
    selected = current_app.extensions.get("selected_clients", {}).get(round_id, [])
    received = current_app.extensions.get("round_updates", {}).get(round_id, {})
    tracked = current_app.extensions.get("update_submission_status", {}).get(round_id, {})
    submitted = sorted(received)
    missing = sorted(set(selected) - set(received))
    expired = sorted(
        client_id
        for client_id, status in tracked.items()
        if status.get("status") == "expired"
    )
    return jsonify(
        round=round_id,
        selected_clients=selected,
        submitted_clients=submitted,
        missing_clients=missing,
        expired_clients=expired,
        updates=received,
        status=tracked,
    )


@aggregation_bp.post("/api/rounds/<int:round_id>/timeout")
def notify_update_timeout(round_id: int):
    """기한 초과 클라이언트에 타임아웃을 알리고 미제출 상태를 저장한다."""
    selected = current_app.extensions.get("selected_clients", {}).get(round_id, [])
    received = current_app.extensions.get("round_updates", {}).get(round_id, {})
    missing = sorted(set(selected) - set(received))
    registered = current_app.extensions.get("clients", {})
    statuses = current_app.extensions.setdefault("update_submission_status", {})
    round_status = statuses.setdefault(round_id, {})
    notified = []

    for client_id in missing:
        round_status.setdefault(client_id, {})["status"] = "expired"
        round_status[client_id]["expired_at"] = datetime.now(timezone.utc).isoformat()
        client = registered.get(client_id)
        if client:
            try:
                _post_json(
                    f"{client['url']}/api/round/timeout",
                    {"round": round_id},
                    timeout=1.0,
                )
                notified.append(client_id)
            except (OSError, ValueError, URLError):
                current_app.logger.warning(
                    "Timeout notification failed: client_id=%s", client_id
                )

    return jsonify(
        round=round_id,
        expired_clients=missing,
        notified_clients=notified,
    )


def _post_json(url: str, payload: dict, timeout: float = 3.0):
    http_request = Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(http_request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


@aggregation_bp.post("/api/rounds/<int:round_id>/aggregate")
def aggregate_round(round_id: int):
    """유효한 클라이언트 업데이트만 채택하고 가상 집계 결과를 기록한다."""
    payload = request.get_json(silent=True) or {}
    updates = payload.get("updates", [])
    accepted = [update for update in updates if update.get("valid", True)]
    if not accepted:
        return jsonify(error="At least one valid update is required"), 400

    result = {
        "round": round_id,
        "accepted_count": len(accepted),
        "aggregation": payload.get("aggregation", "fedavg"),
        "update_ids": [update["client_id"] for update in accepted],
    }
    current_app.extensions["round_results"].append(result)
    return jsonify(result)


@aggregation_bp.post("/api/rounds/<int:round_id>/execute")
@one_round_at_a_time
def execute_round(round_id: int):
    """수신한 클라이언트 업데이트를 유효성 검사하고 단일 라운드를 집계한다."""
    payload = request.get_json(silent=True) or {}
    return execute_result(round_id, payload)


@aggregation_bp.post("/api/rounds/<int:round_id>/start")
@one_round_at_a_time
def start_round(round_id: int):
    """실험 관리자의 시작 신호를 받아 선택 클라이언트 실행과 집계를 처리한다."""
    if round_id in current_app.extensions.get("completed_rounds", set()):
        return jsonify(error="Configure a new experiment before reusing a completed round"), 409
    payload = request.get_json(silent=True) or {}
    config = payload or current_app.extensions.get("round_config", {})
    config["round"] = round_id
    if not current_app.extensions.get("dataset_range", {}).get("ranges"):
        return jsonify(error="Distribute the test dataset before starting"), 409

    registered = list(current_app.extensions.get("clients", {}).values())
    if not registered:
        return jsonify(error="No registered clients"), 409

    availability_responses = []
    available_clients = []
    for client in registered:
        try:
            response = _post_json(
                f"{client['url']}/api/availability",
                {"round": round_id},
            )
            response["client_id"] = response.get("client_id", client["client_id"])
        except (OSError, ValueError):
            response = {
                "client_id": client["client_id"],
                "available": False,
                "reason": "availability_request_failed",
            }
        availability_responses.append(response)
        if response.get("available", False):
            available_clients.append(client)

    current_app.extensions["availability_responses"][round_id] = availability_responses
    current_app.extensions["available_clients"][round_id] = [
        client["client_id"] for client in available_clients
    ]

    if not available_clients:
        return jsonify(error="No available clients"), 409

    # 이전 라운드 제출이 재사용되지 않도록 시작 시 해당 라운드 업데이트를 초기화한다.
    current_app.extensions.setdefault("round_updates", {})[round_id] = {}
    current_app.extensions.setdefault("update_submission_status", {})[round_id] = {}

    seed = config.get("seed", 42)
    participation = config.get("participation_rate", 0.5)
    nums_clients = config.get("nums_clients", len(registered))
    selected_count = max(1, round(nums_clients * participation))
    rng = random.Random(seed + round_id)
    selected = sorted(rng.sample(range(len(available_clients)), min(selected_count, len(available_clients))))
    selected_clients = [available_clients[index] for index in selected]
    selected_ids = [client["client_id"] for client in selected_clients]
    current_app.extensions["selected_clients"][round_id] = selected_ids
    limit = TrainingSettings.from_dict(config).max_train_samples
    counts = {item["client_id"]: item.get("num_samples", 0) for item in availability_responses}
    current_app.extensions.setdefault("expected_sample_counts", {})[round_id] = {
        client_id: min(counts[client_id], limit) if limit else counts[client_id] for client_id in selected_ids
    }
    current_app.extensions["update_submission_status"][round_id] = {
        client_id: {"status": "waiting"}
        for client_id in selected_ids
    }

    timeout_seconds = float(
        config.get("round_timeout", config.get("update_timeout", 900.0))
    )
    app = current_app._get_current_object()
    deadline = datetime.now(timezone.utc) + timedelta(seconds=max(0, timeout_seconds))
    current_app.extensions.setdefault("update_deadlines", {})[round_id] = (
        deadline.isoformat()
    )

    def notify_expired() -> None:
        from flask import current_app as flask_current_app

        with app.app_context():
            if round_id in app.extensions.get("completed_rounds", set()):
                return
            flask_current_app.extensions.setdefault("update_submission_status", {})[
                round_id
            ]
            selected = flask_current_app.extensions["selected_clients"].get(round_id, [])
            received = flask_current_app.extensions["round_updates"].get(round_id, {})
            for client_id in sorted(set(selected) - set(received)):
                status = flask_current_app.extensions[
                    "update_submission_status"
                ][round_id]
                status.setdefault(client_id, {})["status"] = "expired"
                status[client_id]["expired_at"] = datetime.now(timezone.utc).isoformat()
                client = flask_current_app.extensions["clients"].get(client_id)
                if client:
                    try:
                        _post_json(
                            f"{client['url']}/api/round/timeout",
                            {"round": round_id},
                            timeout=1.0,
                        )
                    except (OSError, ValueError, URLError):
                        app.logger.warning(
                            "Timeout notification failed: client_id=%s", client_id
                        )

    timer = threading.Timer(timeout_seconds, notify_expired)
    timer.daemon = True
    current_app.extensions["round_timer"] = timer
    timer.start()

    global_model = current_app.extensions.get("global_model")
    if global_model is None:
        return jsonify(error="Global model is not initialized"), 409
    for client in selected_clients:
        try:
            _post_json(
                f"{client['url']}/api/model/global",
                {**global_model, "round": round_id, "training_config": {**TrainingSettings.from_dict(config).to_dict(), "seed": seed},
                 "update_deadline": deadline.isoformat()},
                timeout=max(0.001, (deadline - datetime.now(timezone.utc)).total_seconds()),
            )
        except (OSError, ValueError):
            current_app.logger.warning(
                "Global model delivery failed: client_id=%s", client["client_id"]
            )

    delay_ids = set(config.get("delay_clients", []))
    accepted_clients = [
        client for client in selected_clients if client["client_id"] not in delay_ids
    ]
    malicious_ids = set(config.get("malicious_clients", []))

    updates = []
    malicious_local_updates = {}
    attack_enabled = bool(config.get("attack_enabled", False))

    for client in accepted_clients:
        try:
            is_malicious = attack_enabled and client["client_id"] in malicious_ids
            local_response = _post_json(
                f"{client['url']}/api/round/execute",
                {
                    "round": round_id,
                    "malicious": is_malicious,
                    "attack_enabled": attack_enabled,
                },
            )
            if is_malicious:
                malicious_local_updates[client["client_id"]] = local_response
                continue
            _post_json(
                f"{client['url']}/api/round/submit",
                {
                    "round": round_id,
                    "malicious": False,
                    "attack_enabled": attack_enabled,
                    "server_url": config.get(
                        "server_url",
                        f"http://{request.host.split(':')[0]}:{current_app.config['PORT']}",
                    ),
                },
            )
        except (OSError, ValueError):
            current_app.logger.warning(
                "Client update generation failed: client_id=%s",
                client["client_id"],
            )

    # 악성 클라이언트들의 추출값을 서로의 API로 직접 공유한다.
    if attack_enabled:
        for client in accepted_clients:
            sender_id = client["client_id"]
            if sender_id not in malicious_local_updates:
                continue
            for peer in accepted_clients:
                peer_id = peer["client_id"]
                if peer_id not in malicious_local_updates or peer_id == sender_id:
                    continue
                try:
                    _post_json(
                        f"{peer['url']}/api/round/update-share",
                        {
                            "round": round_id,
                            "sender_id": sender_id,
                            "update": malicious_local_updates[sender_id],
                        },
                    )
                except (OSError, ValueError):
                    current_app.logger.warning(
                        "Malicious update sharing failed: sender=%s peer=%s",
                        sender_id,
                        peer_id,
                    )

    # 악성 클라이언트들이 서로의 원본 업데이트를 공유한 뒤 평균과 배율 공격을 적용한다.
    for client in accepted_clients:
        client_id = client["client_id"]
        if client_id not in malicious_local_updates:
            continue
        peer_updates = [
            peer_update
            for peer_id, peer_update in malicious_local_updates.items()
            if peer_id != client_id
        ]
        try:
            colluded_update = _post_json(
                f"{client['url']}/api/round/collusion",
                {
                    "round": round_id,
                    "own_update": malicious_local_updates[client_id],
                    "peer_updates": peer_updates,
                },
            )
            _post_json(
                f"{client['url']}/api/round/submit",
                {
                    "round": round_id,
                    "malicious": True,
                    "attack_enabled": attack_enabled,
                    "colluded_update": colluded_update,
                    "server_url": config.get(
                        "server_url",
                        f"http://{request.host.split(':')[0]}:{current_app.config['PORT']}",
                    ),
                },
            )
        except (OSError, ValueError):
            current_app.logger.warning(
                "Malicious collusion failed: client_id=%s", client_id
            )

    received_response = current_app.extensions.get("round_updates", {}).get(round_id, {})
    updates = list(received_response.values())
    if not updates:
        timer.cancel()
        return jsonify(error="No client updates received"), 409

    result_payload = {
        **config,
        "selected_clients": selected_ids,
        "delay_clients": [
            client["client_id"] for client in selected_clients if client["client_id"] in delay_ids
        ],
        "updates": updates,
    }
    timer.cancel()
    return execute_result(round_id, result_payload)


def execute_result(round_id: int, payload: dict):
    """업데이트 검증 → 방어 → 집계 → 평가 → 모델 반영 및 결과 기록 순서로 처리한다."""
    if round_id in current_app.extensions.get("completed_rounds", set()):
        return jsonify(error="Round is already completed"), 409
    initial = current_app.extensions.get("global_model")
    updates = [item for item in payload.get("updates", []) if item.get("valid") is True]
    defense_enabled = bool(payload.get("defense_enabled", payload.get("defense", "none") != "none"))
    defense_type = payload.get("defense", "none")
    try:
        if initial is None:
            raise ValueError("Configure the round before execution")
        validate_updates(initial, updates)
        selected_ids = payload.get("selected_clients", [])
        if any(item["round"] != round_id or item["client_id"] not in selected_ids for item in updates):
            raise ValueError("Update round or participation mismatch")
        experiment_id = initial["experiment_id"]
        if payload.get("experiment_id") != experiment_id:
            raise ValueError("Experiment identity mismatch")
        if defense_enabled and defense_type == "norm_clipping":
            updates = apply_norm_clipping(updates, payload.get("clip_norm", 1.0), global_model=initial)
        elif defense_enabled and defense_type != "coordinate_median":
            raise ValueError("Unsupported defense")
        aggregate = aggregate_coordinate_median if defense_enabled and defense_type == "coordinate_median" else aggregate_fedavg
        aggregated = aggregate(initial, updates)
        updated = aggregated["global_model"]
        evaluation = evaluate_global_model(updated, current_app.extensions.get("dataset_range", {}), payload, round_id)
    except (ValueError, KeyError, TypeError, OSError, RuntimeError) as error:
        return jsonify(error=str(error), status="failed", round=round_id), 400

    # 텐서 원본은 디스크에 한 번 저장하고 HTTP 응답과 화면 로그에는 요약만 남긴다.
    import torch
    root = Path(current_app.config.get("RESULTS_DIR", Path(__file__).resolve().parents[2] / "results"))
    experiment_dir = root / experiment_id
    if not experiment_dir.resolve().is_relative_to(root.resolve()):
        return jsonify(error="Invalid experiment result path"), 400
    checkpoint = experiment_dir / "checkpoints" / f"global_{updated['model_version']:04d}.pt"
    save_model_checkpoint(updated, checkpoint)
    for item in updates:
        path = experiment_dir / "updates" / f"round_{round_id:04d}_client_{item['client_id']}.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"round": round_id, "client_id": item["client_id"], "num_samples": item["num_samples"],
                    "base_model_hash": item["base_model_hash"], "base_model_version": item["base_model_version"],
                    "delta": update_delta(initial, item)}, path)
    initial_state, updated_state = decode_weights(initial["weights"]), decode_weights(updated["weights"])
    delta_norm = weights_norm({key: updated_state[key] - initial_state[key] for key in initial_state})
    model_summary = {key: value for key, value in updated.items() if key != "weights"}
    model_summary["checkpoint"] = str(checkpoint)
    accepted_ids = [item["client_id"] for item in updates]
    malicious_ids = payload.get("malicious_clients", [])
    result = {
        "experiment_id": experiment_id, "round": round_id, "model_version": updated["model_version"],
        "selected_clients": selected_ids, "accepted_clients": accepted_ids,
        "delayed_clients": payload.get("delay_clients", []), "malicious_clients": malicious_ids,
        "malicious_selected": sorted(set(selected_ids) & set(malicious_ids)),
        "malicious_accepted": sorted(set(accepted_ids) & set(malicious_ids)),
        "available_clients": current_app.extensions.get("available_clients", {}).get(round_id, []),
        "attack_enabled": bool(payload.get("attack_enabled", False)),
        "attack_type": "insider_collusion" if payload.get("attack_enabled") else None,
        "attack_applied": any(item["update"].get("attack_applied", False) for item in updates),
        "defense": defense_type, "defense_enabled": defense_enabled,
        "defense_type": defense_type if defense_enabled else None,
        "clip_norm": payload.get("clip_norm", 1.0), "attack_scale": payload.get("attack_scale", -5.0),
        "accuracy": evaluation["accuracy"], "loss": evaluation["loss"],
        "aggregated_delta": delta_norm, "global_model_version": updated["model_version"],
        "initial_model_hash": initial["initial_model_hash"],
        "training_config": TrainingSettings.from_dict(payload).to_dict(),
        "training_datasets": {str(item["client_id"]): {"fingerprint": item.get("dataset_fingerprint"), "num_samples": item["num_samples"]} for item in updates},
        "fedavg_samples": aggregated["total_samples"], "fedavg_client_weights": aggregated["client_weights"],
        "global_model": model_summary, "evaluation": evaluation,
        "client_updates": [{key: value for key, value in item.items() if key not in ("local_model", "update")} |
                           {"update": {key: value for key, value in item["update"].items() if key != "values"}} for item in updates],
        "reason": "model_updated", "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    current_app.extensions["global_model"] = updated
    current_app.extensions.setdefault("global_model_history", []).append({
        "round": round_id, "model": model_summary,
        "total_samples": aggregated["total_samples"], "client_weights": aggregated["client_weights"],
    })
    current_app.extensions.setdefault("evaluation_results", {})[round_id] = evaluation
    current_app.extensions.setdefault("completed_rounds", set()).add(round_id)
    current_app.extensions["round_results"].append(result)
    callback_url = payload.get("experiment_manager_url")
    if callback_url:
        try:
            _post_json(callback_url.rstrip("/") + "/api/results/round", result, timeout=30.0)
        except (OSError, ValueError, URLError):
            current_app.logger.warning("Result callback failed: round=%s", round_id)
    return jsonify(result)
