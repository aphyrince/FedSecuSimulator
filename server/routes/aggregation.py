from datetime import datetime, timedelta, timezone
import json
import random
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from pathlib import Path
from functools import wraps
from urllib.error import URLError
from urllib.request import Request, urlopen

from flask import Blueprint, current_app, jsonify, request

from server.services.fedavg import aggregate_coordinate_median, aggregate_fedavg
from server.services.model_evaluation import evaluate_global_model
from server.services.byzantine_defense import apply_norm_clipping
from common.model import (
    decode_weights,
    positive_int,
    save_model_checkpoint,
    weights_hash,
    weights_norm,
)
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
            lock.release()
    return wrapped


def _cancel_round_timer(app, round_id: int) -> None:
    timer = app.extensions.setdefault("round_timers", {}).pop(round_id, None)
    if timer is not None:
        timer.cancel()


def _finish_round(app, round_id: int) -> None:
    with app.extensions.setdefault("round_completion_lock", threading.Lock()):
        app.extensions.setdefault("completed_rounds", set()).add(round_id)
        if app.extensions.get("active_round_id") == round_id:
            app.extensions["active_round_id"] = None
    _cancel_round_timer(app, round_id)


def _maybe_schedule_round_aggregation(app, round_id: int) -> None:
    with app.app_context():
        extensions = app.extensions
        completion_lock = extensions.setdefault(
            "round_completion_lock", threading.Lock()
        )
        with completion_lock:
            if (
                round_id in extensions.get("completed_rounds", set())
                or round_id in extensions.setdefault(
                    "round_aggregation_started", set()
                )
                or not extensions.setdefault("round_delivery_complete", {}).get(
                    round_id, False
                )
            ):
                return

            expected = set(
                extensions.setdefault("round_submission_clients", {}).get(
                    round_id, set()
                )
            )
            received = set(
                extensions.get("round_updates", {}).get(round_id, {})
            )
            statuses = extensions.get("update_submission_status", {}).get(
                round_id, {}
            )
            terminal = received | {
                client_id
                for client_id, status in statuses.items()
                if status.get("status")
                in {"excluded", "expired", "rejected", "failed"}
            }
            deadline_reached = extensions.setdefault(
                "round_deadline_reached", {}
            ).get(round_id, False)
            if not deadline_reached and not expected.issubset(terminal):
                return

            extensions["round_aggregation_started"].add(round_id)
            payload = deepcopy(
                extensions.setdefault("round_payloads", {}).get(round_id, {})
            )
            updates = extensions.get("round_updates", {}).get(round_id, {})
            payload["updates"] = [
                deepcopy(updates[client_id]) for client_id in sorted(updates)
            ]
            _cancel_round_timer(app, round_id)

    threading.Thread(
        target=_run_round_aggregation,
        args=(app, round_id, payload),
        name=f"aggregate-round-{round_id}",
        daemon=True,
    ).start()


def _run_round_aggregation(app, round_id: int, payload: dict) -> None:
    with app.app_context():
        try:
            response = execute_result(round_id, payload)
            if response.status_code >= 400:
                body = response.get_json(silent=True) or {}
                raise RuntimeError(
                    body.get("error", f"Round aggregation failed ({response.status_code})")
                )
        except Exception as error:
            app.logger.exception("Asynchronous round aggregation failed: round=%s", round_id)
            app.extensions.setdefault("round_progress", {}).update(
                round=round_id,
                phase="failed",
                message=str(error),
            )
            _finish_round(app, round_id)
            payload = {
                "round": round_id,
                "experiment_id": payload.get("experiment_id"),
                "status": "failed",
                "error": str(error),
            }
            callback_url = payload.get("experiment_manager_url")
            if callback_url:
                try:
                    _post_json(
                        callback_url.rstrip("/") + "/api/results/round",
                        payload,
                        timeout=30.0,
                    )
                except (OSError, ValueError, URLError):
                    app.logger.exception(
                        "Round failure callback failed: round=%s", round_id
                    )


def _deliver_round_models(app, round_id: int) -> None:
    with app.app_context():
        config = app.extensions["round_payloads"][round_id]
        selected_ids = app.extensions["selected_clients"][round_id]
        clients = app.extensions["clients"]
        attack_enabled = bool(config.get("attack_enabled", False))
        malicious_ids = set(config.get("malicious_clients", []))
        delay_ids = set(config.get("delay_clients", []))
        selected_clients = [
            clients[client_id]
            for client_id in selected_ids
            if client_id not in delay_ids
        ]
        global_model = app.extensions["global_model"]
        deadline = datetime.fromisoformat(
            app.extensions["update_deadlines"][round_id]
        )
        training_config = {
            **TrainingSettings.from_dict(config).to_dict(),
            "seed": config.get("seed", 42),
        }

        for client_id in delay_ids.intersection(selected_ids):
            app.extensions["update_submission_status"][round_id][client_id].update(
                status="excluded",
                reason="configured_delay",
                excluded_at=datetime.now(timezone.utc).isoformat(),
            )

        def deliver(client):
            client_id = client["client_id"]
            return _post_json(
                f"{client['url']}/api/model/global",
                {
                    **global_model,
                    "round": round_id,
                    "attack_enabled": attack_enabled,
                    "server_url": config["server_url"],
                    "training_config": training_config,
                    "update_deadline": deadline.isoformat(),
                    "round_context": {
                        "selected": True,
                        "malicious": attack_enabled and client_id in malicious_ids,
                        "delayed": client_id in delay_ids,
                    },
                },
                timeout=max(
                    0.001,
                    (deadline - datetime.now(timezone.utc)).total_seconds(),
                ),
            )

        failed_deliveries = []
        with ThreadPoolExecutor(max_workers=max(1, len(selected_clients))) as executor:
            futures = {
                executor.submit(deliver, client): client
                for client in selected_clients
            }
            for future in as_completed(futures):
                client_id = futures[future]["client_id"]
                try:
                    future.result()
                except (OSError, ValueError, TypeError) as error:
                    failed_deliveries.append(client_id)
                    app.logger.warning(
                        "Global model delivery failed: round=%s client_id=%s error=%s",
                        round_id,
                        client_id,
                        error,
                    )

        with app.extensions["round_completion_lock"]:
            expected = app.extensions["round_submission_clients"][round_id]
            for client_id in failed_deliveries:
                expected.discard(client_id)
                app.extensions["update_submission_status"][round_id][client_id] = {
                    "status": "excluded",
                    "reason": "model_delivery_failed",
                    "excluded_at": datetime.now(timezone.utc).isoformat(),
                }
            app.extensions["round_delivery_complete"][round_id] = True
            if app.extensions["round_deadline_reached"][round_id]:
                received = app.extensions["round_updates"].get(round_id, {})
                for client_id in expected - set(received):
                    app.extensions["update_submission_status"][round_id][
                        client_id
                    ]["status"] = "expired"

        app.extensions["round_progress"].update(
            phase="waiting_for_updates",
            message="모델 전달이 완료되었습니다. 클라이언트 업데이트를 기다리고 있습니다.",
        )
        _maybe_schedule_round_aggregation(app, round_id)


@aggregation_bp.post("/api/rounds/<int:round_id>/updates")
def receive_client_update(round_id: int):
    """클라이언트가 제출한 라운드 업데이트를 검증해 라운드별로 저장한다."""
    payload = request.get_json(silent=True) or {}
    statuses = current_app.extensions.setdefault("update_submission_status", {})
    round_status = statuses.setdefault(round_id, {})
    attempts = current_app.extensions.setdefault("update_submission_attempts", {})
    round_attempts = attempts.setdefault(round_id, [])

    def reject(reason: str, status_code: int):
        client_id = payload.get("client_id")
        now = datetime.now(timezone.utc).isoformat()
        round_attempts.append({
            "client_id": client_id,
            "status": "rejected",
            "reason": reason,
            "timestamp": now,
        })
        if type(client_id) is int and client_id >= 0:
            if round_status.get(client_id, {}).get("status") != "received":
                previous = round_status.get(client_id, {})
                round_status[client_id] = {
                    **previous,
                    "status": "rejected",
                    "reason": reason,
                    "rejected_at": now,
                }
        return jsonify(error=reason), status_code

    if round_id in current_app.extensions.get("completed_rounds", set()):
        return reject("Round is already completed", 409)
    deadline = current_app.extensions.get("update_deadlines", {}).get(round_id)
    if deadline and datetime.now(timezone.utc) >= datetime.fromisoformat(deadline):
        return reject("Submission deadline expired", 408)
    try:
        positive_int(payload.get("round"), "round")
        positive_int(payload.get("client_id"), "client_id", 0)
    except ValueError as error:
        return reject(str(error), 400)
    if payload.get("round", round_id) != round_id:
        return reject("Round number mismatch", 400)
    client_id = payload.get("client_id")
    if client_id is None or not payload.get("valid", False):
        return reject("Invalid client update", 400)
    selected = current_app.extensions.get("selected_clients", {}).get(round_id, [])
    if int(client_id) not in selected:
        return reject("Client was not selected for this round", 403)
    if (
        round_id in current_app.extensions.get("round_submission_clients", {})
        and int(client_id)
        not in current_app.extensions["round_submission_clients"][round_id]
    ):
        return reject("Client is not expected to submit in this round", 403)
    num_samples = payload.get("num_samples")
    if not isinstance(num_samples, (int, float)) or num_samples < 0:
        return reject("num_samples must be a non-negative number", 400)
    if "update" not in payload or not isinstance(payload["update"], dict):
        return reject("Invalid update payload", 400)
    try:
        update_delta(current_app.extensions["global_model"], payload)
        expected = current_app.extensions.get("expected_sample_counts", {}).get(round_id, {}).get(int(client_id))
        if expected is not None and num_samples != expected:
            raise ValueError("Submitted sample count does not match assigned data")
    except (ValueError, KeyError, TypeError) as error:
        return reject(str(error), 400)

    completion_lock = current_app.extensions.setdefault(
        "round_completion_lock", threading.Lock()
    )
    with completion_lock:
        if (
            round_id in current_app.extensions.get("completed_rounds", set())
            or round_id
            in current_app.extensions.get("round_aggregation_started", set())
        ):
            return reject("Round aggregation has already started", 409)
        deadline = current_app.extensions.get("update_deadlines", {}).get(round_id)
        if deadline and datetime.now(timezone.utc) >= datetime.fromisoformat(deadline):
            return reject("Submission deadline expired", 408)
        if round_status.get(int(client_id), {}).get("status") in {
            "excluded",
            "expired",
            "failed",
        }:
            return reject("Client submission is already closed", 409)
        updates = current_app.extensions.setdefault("round_updates", {})
        round_updates = updates.setdefault(round_id, {})
        if int(client_id) in round_updates:
            return reject("Duplicate client update", 409)
        round_updates[int(client_id)] = payload
        received_at = datetime.now(timezone.utc).isoformat()
        round_status[int(client_id)] = {
            **round_status.get(int(client_id), {}),
            "status": "received",
            "num_samples": num_samples,
            "received_at": received_at,
        }
        round_attempts.append({
            "client_id": int(client_id),
            "status": "received",
            "num_samples": num_samples,
            "timestamp": received_at,
        })
    _maybe_schedule_round_aggregation(
        current_app._get_current_object(), round_id
    )
    return jsonify(
        status="received",
        round=round_id,
        client_id=int(client_id),
        count=len(round_updates),
    )


@aggregation_bp.post("/api/rounds/<int:round_id>/failures")
def receive_client_round_failure(round_id: int):
    """클라이언트가 로컬 학습 또는 제출 실패를 보고한다."""
    payload = request.get_json(silent=True) or {}
    client_id = payload.get("client_id")
    if type(client_id) is not int or client_id < 0:
        return jsonify(error="client_id must be a non-negative integer"), 400
    if payload.get("round", round_id) != round_id:
        return jsonify(error="Round number mismatch"), 400
    if client_id not in current_app.extensions.get("selected_clients", {}).get(
        round_id, []
    ):
        return jsonify(error="Client was not selected for this round"), 403
    expected = current_app.extensions.get("round_submission_clients", {}).get(
        round_id
    )
    if expected is not None and client_id not in expected:
        return jsonify(error="Client is not expected to submit in this round"), 403

    completion_lock = current_app.extensions.setdefault(
        "round_completion_lock", threading.Lock()
    )
    with completion_lock:
        if (
            round_id in current_app.extensions.get("completed_rounds", set())
            or round_id
            in current_app.extensions.get("round_aggregation_started", set())
        ):
            return jsonify(error="Round aggregation has already started"), 409
        deadline = current_app.extensions.get("update_deadlines", {}).get(round_id)
        if deadline and datetime.now(timezone.utc) >= datetime.fromisoformat(deadline):
            return jsonify(error="Submission deadline expired"), 408
        if client_id in current_app.extensions.get("round_updates", {}).get(
            round_id, {}
        ):
            return jsonify(error="Client update was already received"), 409
        now = datetime.now(timezone.utc).isoformat()
        reason = str(payload.get("reason", "client_failure"))
        status = current_app.extensions.setdefault(
            "update_submission_status", {}
        ).setdefault(round_id, {})
        status[client_id] = {
            **status.get(client_id, {}),
            "status": "excluded",
            "reason": reason,
            "error": str(payload.get("error", "")),
            "excluded_at": now,
        }
        current_app.extensions.setdefault(
            "update_submission_attempts", {}
        ).setdefault(round_id, []).append({
            "client_id": client_id,
            "status": "failed",
            "reason": reason,
            "error": str(payload.get("error", "")),
            "timestamp": now,
        })

    _maybe_schedule_round_aggregation(
        current_app._get_current_object(), round_id
    )
    return jsonify(status="recorded", round=round_id, client_id=client_id)


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
    registered = current_app.extensions.get("clients", {})
    statuses = current_app.extensions.setdefault("update_submission_status", {})
    round_status = statuses.setdefault(round_id, {})
    missing = sorted(
        client_id
        for client_id in set(selected) - set(received)
        if round_status.get(client_id, {}).get("reason") != "configured_delay"
    )
    notified = []

    for client_id in missing:
        round_status.setdefault(client_id, {}).update(
            status="expired",
            expired_at=datetime.now(timezone.utc).isoformat(),
        )
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
    active_round = current_app.extensions.get("active_round_id")
    if active_round is not None and active_round not in current_app.extensions.get(
        "completed_rounds", set()
    ):
        return jsonify(error="An asynchronous round is still active"), 409
    payload = request.get_json(silent=True) or {}
    return execute_result(round_id, payload)


@aggregation_bp.post("/api/rounds/<int:round_id>/start")
@one_round_at_a_time
def start_round(round_id: int):
    """실험 관리자의 시작 신호를 받아 선택 클라이언트 실행과 집계를 처리한다."""
    if round_id in current_app.extensions.get("completed_rounds", set()):
        return jsonify(error="Configure a new experiment before reusing a completed round"), 409
    active_round = current_app.extensions.get("active_round_id")
    if active_round is not None and active_round not in current_app.extensions.get(
        "completed_rounds", set()
    ):
        return jsonify(error=f"Round {active_round} is still active"), 409
    payload = request.get_json(silent=True) or {}
    config = payload or current_app.extensions.get("round_config", {})
    config["round"] = round_id
    progress = current_app.extensions.setdefault(
        "round_progress", {"round": None, "phase": "idle"}
    )
    progress.update(
        round=round_id,
        phase="checking_availability",
        message="클라이언트 참여 가능 여부를 확인하고 있습니다.",
        config={
            key: config.get(key)
            for key in (
                "participation_rate",
                "attack_enabled",
                "malicious_clients",
                "delay_clients",
                "defense_enabled",
                "defense",
                "max_train_samples",
            )
        },
        started_at=datetime.now(timezone.utc).isoformat(),
    )
    if not current_app.extensions.get("dataset_range", {}).get("ranges"):
        progress.update(phase="failed", message="테스트 데이터가 분배되지 않았습니다.")
        return jsonify(error="Distribute the test dataset before starting"), 409

    registered = list(current_app.extensions.get("clients", {}).values())
    if not registered:
        progress.update(phase="failed", message="등록된 클라이언트가 없습니다.")
        return jsonify(error="No registered clients"), 409
    global_model = current_app.extensions.get("global_model")
    if global_model is None:
        progress.update(phase="failed", message="전역 모델이 초기화되지 않았습니다.")
        return jsonify(error="Global model is not initialized"), 409

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
        response["checked_at"] = datetime.now(timezone.utc).isoformat()
        availability_responses.append(response)
        if response.get("available", False):
            available_clients.append(client)

    current_app.extensions["availability_responses"][round_id] = availability_responses
    current_app.extensions["available_clients"][round_id] = [
        client["client_id"] for client in available_clients
    ]

    if not available_clients:
        current_app.extensions["selected_clients"][round_id] = []
        current_app.extensions["round_updates"][round_id] = {}
        current_app.extensions["update_submission_status"][round_id] = {}
        current_app.extensions["update_submission_attempts"][round_id] = []
        current_app.extensions["round_submission_clients"][round_id] = set()
        current_app.extensions["round_delivery_complete"][round_id] = True
        current_app.extensions["round_deadline_reached"][round_id] = False
        current_app.extensions["round_payloads"][round_id] = {
            **config,
            "selected_clients": [],
            "delay_clients": [],
            "experiment_manager_url": config.get("experiment_manager_url"),
        }
        current_app.extensions["active_round_id"] = round_id
        progress.update(
            phase="waiting_for_updates",
            message="참여 가능한 클라이언트가 없어 결과를 마무리하고 있습니다.",
        )
        _maybe_schedule_round_aggregation(
            current_app._get_current_object(), round_id
        )
        return jsonify(status="started", round=round_id, selected_clients=[]), 202

    # 이전 라운드 제출이 재사용되지 않도록 시작 시 해당 라운드 업데이트를 초기화한다.
    current_app.extensions.setdefault("round_updates", {})[round_id] = {}
    current_app.extensions.setdefault("update_submission_status", {})[round_id] = {}
    current_app.extensions.setdefault("update_submission_attempts", {})[round_id] = []

    seed = config.get("seed", 42)
    participation = config.get("participation_rate", 0.5)
    nums_clients = config.get("nums_clients", len(registered))
    selected_count = max(1, round(nums_clients * participation))
    rng = random.Random(seed + round_id)
    selected = sorted(rng.sample(range(len(available_clients)), min(selected_count, len(available_clients))))
    selected_clients = [available_clients[index] for index in selected]
    selected_ids = [client["client_id"] for client in selected_clients]
    current_app.extensions["selected_clients"][round_id] = selected_ids
    selected_at = datetime.now(timezone.utc).isoformat()
    limit = TrainingSettings.from_dict(config).max_train_samples
    counts = {item["client_id"]: item.get("num_samples", 0) for item in availability_responses}
    current_app.extensions.setdefault("expected_sample_counts", {})[round_id] = {
        client_id: min(counts[client_id], limit) if limit else counts[client_id] for client_id in selected_ids
    }
    attack_enabled = bool(config.get("attack_enabled", False))
    malicious_ids = set(config.get("malicious_clients", []))
    delay_ids = set(config.get("delay_clients", []))
    current_app.extensions["update_submission_status"][round_id] = {
        client_id: {
            "status": "waiting",
            "selected_at": selected_at,
            "delayed": client_id in delay_ids,
        }
        for client_id in selected_ids
    }
    progress.update(
        phase="training",
        message="선정된 클라이언트가 로컬 학습을 진행하고 있습니다.",
        available_clients=current_app.extensions["available_clients"][round_id],
        selected_clients=selected_ids,
        malicious_clients=sorted(malicious_ids.intersection(selected_ids))
        if attack_enabled
        else [],
        delayed_clients=sorted(delay_ids.intersection(selected_ids)),
    )

    timeout_seconds = float(
        config.get("round_timeout", config.get("update_timeout", 900.0))
    )
    app = current_app._get_current_object()
    deadline = datetime.now(timezone.utc) + timedelta(seconds=max(0, timeout_seconds))
    current_app.extensions.setdefault("update_deadlines", {})[round_id] = (
        deadline.isoformat()
    )
    delay_clients = [
        client["client_id"]
        for client in selected_clients
        if client["client_id"] in delay_ids
    ]
    current_app.extensions["round_payloads"][round_id] = {
        **config,
        "selected_clients": selected_ids,
        "delay_clients": delay_clients,
        "server_url": config.get(
            "server_url",
            f"http://{request.host.split(':')[0]}:{current_app.config['PORT']}",
        ),
        "experiment_manager_url": config.get("experiment_manager_url"),
    }
    current_app.extensions["round_submission_clients"][round_id] = (
        set(selected_ids) - set(delay_clients)
    )
    current_app.extensions["round_delivery_complete"][round_id] = False
    current_app.extensions["round_deadline_reached"][round_id] = False
    current_app.extensions["round_aggregation_started"].discard(round_id)
    current_app.extensions["active_round_id"] = round_id

    def notify_expired() -> None:
        with app.app_context():
            with app.extensions["round_completion_lock"]:
                if (
                    round_id in app.extensions.get("completed_rounds", set())
                    or round_id
                    in app.extensions.get("round_aggregation_started", set())
                ):
                    return
                app.extensions["round_deadline_reached"][round_id] = True
                expected = app.extensions["round_submission_clients"].get(
                    round_id, set()
                )
                received = app.extensions["round_updates"].get(round_id, {})
                statuses = app.extensions["update_submission_status"][round_id]
                missing = sorted(set(expected) - set(received))
                for client_id in missing:
                    statuses.setdefault(client_id, {}).update(
                        status="expired",
                        expired_at=datetime.now(timezone.utc).isoformat(),
                    )

            for client_id in missing:
                client = app.extensions["clients"].get(client_id)
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
            _maybe_schedule_round_aggregation(app, round_id)

    timer = threading.Timer(timeout_seconds, notify_expired)
    timer.daemon = True
    current_app.extensions["round_timers"][round_id] = timer
    timer.start()

    threading.Thread(
        target=_deliver_round_models,
        args=(app, round_id),
        name=f"deliver-round-{round_id}",
        daemon=True,
    ).start()
    return jsonify(
        status="started",
        round=round_id,
        selected_clients=selected_ids,
        expected_clients=sorted(
            current_app.extensions["round_submission_clients"][round_id]
        ),
        deadline=deadline.isoformat(),
    ), 202


def execute_result(round_id: int, payload: dict):
    """업데이트 검증 → 방어 → 집계 → 평가 → 모델 반영 및 결과 기록 순서로 처리한다."""
    if round_id in current_app.extensions.get("completed_rounds", set()):
        return jsonify(error="Round is already completed"), 409
    initial = current_app.extensions.get("global_model")
    updates = [item for item in payload.get("updates", []) if item.get("valid") is True]
    defense_enabled = bool(payload.get("defense_enabled", payload.get("defense", "none") != "none"))
    defense_type = payload.get("defense", "none")
    progress = current_app.extensions.setdefault(
        "round_progress", {"round": round_id, "phase": "idle"}
    )
    progress.update(round=round_id, phase="aggregating", message="업데이트를 집계하고 있습니다.")
    try:
        if initial is None:
            raise ValueError("Configure the round before execution")
        selected_ids = payload.get("selected_clients", [])
        if any(item["round"] != round_id or item["client_id"] not in selected_ids for item in updates):
            raise ValueError("Update round or participation mismatch")
        experiment_id = initial["experiment_id"]
        if payload.get("experiment_id") != experiment_id:
            raise ValueError("Experiment identity mismatch")
        if updates:
            validate_updates(initial, updates)
        root = Path(current_app.config.get("RESULTS_DIR", Path(__file__).resolve().parents[2] / "results"))
        experiment_dir = root / experiment_id
        if not experiment_dir.resolve().is_relative_to(root.resolve()):
            return jsonify(error="Invalid experiment result path"), 400
        start_checkpoint = experiment_dir / "checkpoints" / f"round_{round_id:04d}_start.pt"
        save_model_checkpoint(initial, start_checkpoint)

        status_by_client = current_app.extensions.get("update_submission_status", {}).get(round_id, {})
        accepted_ids = sorted(item["client_id"] for item in updates)
        available_ids = current_app.extensions.get("available_clients", {}).get(round_id, [])
        availability = {
            item.get("client_id"): item
            for item in current_app.extensions.get("availability_responses", {}).get(round_id, [])
        }
        received_by_client = {item["client_id"]: item for item in updates}
        closed_at = datetime.now(timezone.utc).isoformat()
        client_activity = []
        for client_id in sorted(
            set(available_ids) | set(selected_ids) | set(status_by_client) | set(availability)
        ):
            submission = status_by_client.get(client_id, {})
            update = received_by_client.get(client_id, {})
            is_accepted = client_id in accepted_ids
            if is_accepted:
                status, reason = "accepted", None
                accepted_at = datetime.now(timezone.utc).isoformat()
                submission.update(status="accepted", accepted_at=accepted_at)
            elif submission.get("status") == "rejected":
                status, reason = "excluded", submission.get("reason", "update_rejected")
                accepted_at = None
            elif submission.get("status") == "excluded":
                status, reason = "excluded", submission.get("reason", "excluded")
                accepted_at = None
            elif submission.get("status") == "expired":
                status, reason = "excluded", "submission_timeout"
                accepted_at = None
            elif client_id not in available_ids:
                status, reason = "excluded", availability.get(client_id, {}).get("reason", "unavailable")
                accepted_at = None
            elif client_id not in selected_ids:
                status, reason = "excluded", "not_selected"
                accepted_at = None
            else:
                status, reason = "excluded", "no_update_received"
                accepted_at = None
            client_activity.append({
                "client_id": client_id,
                "status": status,
                "reason": reason,
                "selected": client_id in selected_ids,
                "delayed": bool(
                    submission.get(
                        "delayed",
                        client_id in selected_ids
                        and client_id in payload.get("delay_clients", []),
                    )
                ),
                "selected_at": submission.get("selected_at"),
                "accepted_at": accepted_at,
                "submission_status": submission.get("status"),
                "malicious": bool(payload.get("attack_enabled")) and client_id in payload.get("malicious_clients", []),
                "num_samples": update.get(
                    "num_samples",
                    submission.get(
                        "num_samples", availability.get(client_id, {}).get("num_samples")
                    ),
                ),
                "training_accuracy": update.get("training_accuracy"),
                "training_started_at": update.get("training_started_at"),
                "training_completed_at": update.get("training_completed_at"),
                "training_duration_seconds": update.get("training_duration_seconds"),
                "received_at": submission.get("received_at"),
                "submitted_at": submission.get("received_at"),
                "excluded_at": (
                    submission.get("rejected_at")
                    or submission.get("expired_at")
                    or submission.get("excluded_at")
                    or availability.get(client_id, {}).get("checked_at")
                    or (None if is_accepted else closed_at)
                ),
                "update_norm_before_defense": update.get("update", {}).get("norm_before_defense", update.get("update", {}).get("update_norm")),
                "update_norm_after_defense": update.get("update", {}).get("norm_after_defense"),
            })

        malicious_ids = payload.get("malicious_clients", [])
        malicious_selected = sorted(set(selected_ids) & set(malicious_ids))
        malicious_accepted = sorted(set(accepted_ids) & set(malicious_ids))
        population_size = max(1, int(payload.get("nums_clients", len(available_ids) or len(selected_ids))))
        malicious_ratio = len(set(malicious_ids)) / population_size
        accepted_malicious_ratio = (
            len(malicious_accepted) / len(accepted_ids) if accepted_ids else "N/A"
        )
        common_result = {
            "experiment_id": experiment_id,
            "round": round_id,
            "model_version": initial["model_version"],
            "selected_clients": selected_ids,
            "accepted_clients": accepted_ids,
            "available_clients": available_ids,
            "malicious_clients": malicious_ids,
            "malicious_selected": malicious_selected,
            "malicious_accepted": malicious_accepted,
            "malicious_ratio": malicious_ratio,
            "accepted_malicious_ratio": accepted_malicious_ratio,
            "client_activity": client_activity,
            "submission_attempts": current_app.extensions.get(
                "update_submission_attempts", {}
            ).get(round_id, []),
            "attack_enabled": bool(payload.get("attack_enabled", False)),
            "attack_type": "insider_collusion" if payload.get("attack_enabled") else None,
            "defense": payload.get("defense", "none"),
            "defense_enabled": defense_enabled,
            "defense_type": payload.get("defense") if defense_enabled else None,
            "training_config": TrainingSettings.from_dict(payload).to_dict(),
            "initial_model_hash": initial["initial_model_hash"],
            "start_model_hash": weights_hash(initial["weights"]),
            "start_checkpoint": str(start_checkpoint),
            "round_config": {
                key: value
                for key, value in payload.items()
                if key not in {"updates", "initial_global_model"}
            },
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        if len(accepted_ids) < 3:
            progress.update(phase="evaluating", message="채택 업데이트가 3개 미만이어서 모델을 갱신하지 않습니다.")
            evaluation_error = None
            try:
                evaluation = evaluate_global_model(
                    initial,
                    current_app.extensions.get("dataset_range", {}),
                    payload,
                    round_id,
                )
            except (ValueError, KeyError, TypeError, OSError, RuntimeError) as error:
                evaluation = None
                evaluation_error = str(error)
            result = {
                **common_result,
                "accuracy": evaluation["accuracy"] if evaluation else None,
                "loss": evaluation["loss"] if evaluation else None,
                "evaluation": evaluation,
                "evaluation_error": evaluation_error,
                "training_datasets": {
                    str(item["client_id"]): {
                        "fingerprint": item.get("dataset_fingerprint"),
                        "num_samples": item["num_samples"],
                    }
                    for item in updates
                },
                "global_model_version": initial["model_version"],
                "global_model": {
                    key: value for key, value in initial.items() if key != "weights"
                } | {"checkpoint": str(start_checkpoint)},
                "model_updated": False,
                "aggregated_delta": 0.0,
                "defense_aggregate_norm": 0.0,
                "fedavg_samples": sum(item["num_samples"] for item in updates),
                "client_updates": [
                    {
                        **{
                            key: value
                            for key, value in item.items()
                            if key not in {"local_model", "update"}
                        },
                        "update": {
                            key: value
                            for key, value in item.get("update", {}).items()
                            if key != "values"
                        },
                    }
                    for item in updates
                ],
                "delayed_clients": payload.get("delay_clients", []),
                "clip_norm": payload.get("clip_norm", 1.0),
                "attack_scale": payload.get("attack_scale", -5.0),
                "reason": "insufficient_accepted_updates",
            }
            if evaluation is not None:
                current_app.extensions.setdefault("evaluation_results", {})[round_id] = evaluation
                _finish_round(current_app._get_current_object(), round_id)
            current_app.extensions["round_results"].append(result)
            progress.update(
                phase="completed",
                message=(
                    "3개 미만의 업데이트로 모델을 갱신하지 않았습니다."
                    if evaluation is not None
                    else f"모델을 갱신하지 않았으며 정확도 평가에 실패했습니다: {evaluation_error}"
                ),
            )
            callback_url = payload.get("experiment_manager_url")
            if callback_url:
                try:
                    _post_json(callback_url.rstrip("/") + "/api/results/round", result, timeout=30.0)
                except (OSError, ValueError, URLError):
                    current_app.logger.warning("Result callback failed: round=%s", round_id)
            return jsonify(result)

        for item in updates:
            item.setdefault("update", {})["norm_before_defense"] = weights_norm(update_delta(initial, item))
        if defense_enabled and defense_type == "norm_clipping":
            updates = apply_norm_clipping(updates, payload.get("clip_norm", 1.0), global_model=initial)
        elif defense_enabled and defense_type != "coordinate_median":
            raise ValueError("Unsupported defense")
        for item in updates:
            item.setdefault("update", {})["norm_after_defense"] = (
                weights_norm(update_delta(initial, item))
                if defense_enabled and defense_type == "norm_clipping"
                else None
                if defense_enabled and defense_type == "coordinate_median"
                else item["update"].get("norm_before_defense")
            )
        activity_by_client = {
            entry["client_id"]: entry for entry in client_activity
        }
        for item in updates:
            activity = activity_by_client[item["client_id"]]
            activity.update(
                update_norm_before_attack=item["update"].get("norm_before_attack"),
                update_norm_after_attack=item["update"].get("norm_after_attack"),
                update_norm_before_defense=item["update"].get("norm_before_defense"),
                update_norm_after_defense=item["update"].get("norm_after_defense"),
            )
        aggregate = aggregate_coordinate_median if defense_enabled and defense_type == "coordinate_median" else aggregate_fedavg
        aggregated = aggregate(initial, updates)
        updated = aggregated["global_model"]
        progress.update(phase="evaluating", message="전역 모델을 평가하고 있습니다.")
        evaluation = evaluate_global_model(updated, current_app.extensions.get("dataset_range", {}), payload, round_id)
    except (ValueError, KeyError, TypeError, OSError, RuntimeError) as error:
        progress.update(phase="failed", message=str(error))
        return jsonify(error=str(error), status="failed", round=round_id), 400

    # 텐서 원본은 디스크에 한 번 저장하고 HTTP 응답과 화면 로그에는 요약만 남긴다.
    import torch
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
    result = {
        **common_result,
        "model_version": updated["model_version"],
        "selected_clients": selected_ids, "accepted_clients": accepted_ids,
        "delayed_clients": payload.get("delay_clients", []),
        "attack_applied": any(item["update"].get("attack_applied", False) for item in updates),
        "clip_norm": payload.get("clip_norm", 1.0), "attack_scale": payload.get("attack_scale", -5.0),
        "accuracy": evaluation["accuracy"], "loss": evaluation["loss"],
        "aggregated_delta": delta_norm, "defense_aggregate_norm": delta_norm,
        "global_model_version": updated["model_version"],
        "initial_model_hash": initial["initial_model_hash"],
        "training_config": TrainingSettings.from_dict(payload).to_dict(),
        "training_datasets": {str(item["client_id"]): {"fingerprint": item.get("dataset_fingerprint"), "num_samples": item["num_samples"]} for item in updates},
        "fedavg_samples": aggregated["total_samples"], "fedavg_client_weights": aggregated["client_weights"],
        "global_model": model_summary, "evaluation": evaluation,
        "client_updates": [{key: value for key, value in item.items() if key not in ("local_model", "update")} |
                           {"update": {key: value for key, value in item["update"].items() if key != "values"}} for item in updates],
        "model_updated": True,
        "reason": "model_updated",
    }
    current_app.extensions["global_model"] = updated
    current_app.extensions.setdefault("global_model_history", []).append({
        "round": round_id, "model": model_summary,
        "total_samples": aggregated["total_samples"], "client_weights": aggregated["client_weights"],
    })
    current_app.extensions.setdefault("evaluation_results", {})[round_id] = evaluation
    _finish_round(current_app._get_current_object(), round_id)
    current_app.extensions["round_results"].append(result)
    progress.update(phase="completed", message="라운드 집계와 평가가 완료되었습니다.")
    callback_url = payload.get("experiment_manager_url")
    if callback_url:
        try:
            _post_json(callback_url.rstrip("/") + "/api/results/round", result, timeout=30.0)
        except (OSError, ValueError, URLError):
            current_app.logger.warning("Result callback failed: round=%s", round_id)
    return jsonify(result)
