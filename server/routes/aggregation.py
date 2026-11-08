from datetime import datetime, timedelta, timezone
import json
import random
import threading
from copy import deepcopy
from urllib.error import URLError
from urllib.request import Request, urlopen

from flask import Blueprint, current_app, jsonify, request

from services.fedavg import aggregate_fedavg
from services.model_evaluation import evaluate_global_model


aggregation_bp = Blueprint("server_aggregation", __name__)


@aggregation_bp.post("/api/rounds/<int:round_id>/updates")
def receive_client_update(round_id: int):
    """클라이언트가 제출한 라운드 업데이트를 검증해 라운드별로 저장한다."""
    payload = request.get_json(silent=True) or {}
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

    updates = current_app.extensions.setdefault("round_updates", {})
    round_updates = updates.setdefault(round_id, {})
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
def execute_round(round_id: int):
    """수신한 클라이언트 업데이트를 유효성 검사하고 단일 라운드를 집계한다."""
    payload = request.get_json(silent=True) or {}
    return execute_result(round_id, payload)


@aggregation_bp.post("/api/rounds/<int:round_id>/start")
def start_round(round_id: int):
    """실험 관리자의 시작 신호를 받아 선택 클라이언트 실행과 집계를 처리한다."""
    payload = request.get_json(silent=True) or {}
    config = payload or current_app.extensions.get("round_config", {})
    config["round"] = round_id

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
    current_app.extensions["update_submission_status"][round_id] = {
        client_id: {"status": "waiting"}
        for client_id in selected_ids
    }

    timeout_seconds = float(
        config.get("round_timeout", config.get("update_timeout", 60.0))
    )
    app = current_app._get_current_object()
    deadline = datetime.now(timezone.utc) + timedelta(seconds=max(0, timeout_seconds))
    current_app.extensions.setdefault("update_deadlines", {})[round_id] = (
        deadline.isoformat()
    )

    def notify_expired() -> None:
        from flask import current_app as flask_current_app

        with app.app_context():
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
    timer.start()

    global_model = current_app.extensions.get("global_model")
    if global_model is None:
        return jsonify(error="Global model is not initialized"), 409
    for client in selected_clients:
        try:
            _post_json(
                f"{client['url']}/api/model/global",
                {**global_model, "round": round_id},
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
    for client in accepted_clients:
        try:
            updates.append(
                _post_json(
                    f"{client['url']}/api/round/submit",
                    {
                        "round": round_id,
                        "malicious": client["client_id"] in malicious_ids,
                        "attack_enabled": config.get("attack_enabled", False),
                        "server_url": config.get(
                            "server_url",
                            f"http://{request.host.split(':')[0]}:{current_app.config['PORT']}",
                        ),
                    },
                )
            )
        except (OSError, ValueError):
            current_app.logger.warning("Client update failed: client_id=%s", client["client_id"])

    received_response = current_app.extensions.get("round_updates", {}).get(round_id, {})
    updates = list(received_response.values())
    if not updates:
        return jsonify(error="No client updates received"), 409

    result_payload = {
        **config,
        "selected_clients": selected_ids,
        "delay_clients": [
            client["client_id"] for client in selected_clients if client["client_id"] in delay_ids
        ],
        "updates": updates,
    }
    return execute_result(round_id, result_payload)


def execute_result(round_id: int, payload: dict):
    updates = [update for update in payload.get("updates", []) if update.get("valid", True)]
    if not updates:
        return jsonify(error="At least one valid update is required"), 400

    delta_values = [
        update["update"]["model_delta"]
        for update in updates
        if isinstance(update["update"].get("model_delta"), (int, float))
    ]
    if delta_values:
        values = delta_values
    else:
        values = [
            update["update"].get("update_norm", 0.0)
            for update in updates
        ]
    if payload.get("defense") == "coordinate_median":
        values.sort()
        aggregated_delta = values[len(values) // 2]
    else:
        aggregated_delta = sum(values) / len(values)

    attack = payload.get("attack_enabled", False)
    selected_ids = payload.get("selected_clients", [])
    accepted_ids = [update["client_id"] for update in updates]
    malicious_ids = payload.get("malicious_clients", [])
    try:
        fedavg_result = aggregate_fedavg(
            current_app.extensions.get("global_model", {}),
            updates,
        )
        current_app.extensions["global_model"] = fedavg_result["global_model"]
        current_app.extensions.setdefault("global_model_history", []).append({
            "round": round_id,
            "model": deepcopy(fedavg_result["global_model"]),
            "total_samples": fedavg_result["total_samples"],
            "client_weights": fedavg_result["client_weights"],
        })
    except ValueError as exc:
        current_app.logger.warning("FedAvg update skipped: %s", exc)
        fedavg_result = {
            "total_samples": 0,
            "client_weights": {},
        }

    try:
        evaluation = evaluate_global_model(
            global_model=current_app.extensions.get("global_model", {}),
            dataset_range=current_app.extensions.get("dataset_range", {}),
            evaluation_config=payload,
            round_id=round_id,
        )
        current_app.extensions["evaluation_results"][round_id] = evaluation
    except (ValueError, TypeError, KeyError) as exc:
        current_app.logger.warning(
            "Global model evaluation failed: round=%s error=%s",
            round_id,
            exc,
        )
        evaluation = {
            "accuracy": 0.0,
            "loss": 1.0,
            "num_test_samples": 0,
            "correct_predictions": 0,
            "metrics": {},
        }
        current_app.extensions["evaluation_results"][round_id] = evaluation

    accuracy = evaluation["accuracy"]
    loss = evaluation["loss"]

    result = {
        "experiment_id": payload.get("experiment_id", "unknown"),
        "round": round_id,
        "model_version": round_id,
        "selected_clients": payload.get("selected_clients", []),
        "accepted_clients": accepted_ids,
        "delayed_clients": payload.get("delay_clients", []),
        "malicious_clients": malicious_ids,
        "malicious_selected": [
            client_id for client_id in selected_ids if client_id in malicious_ids
        ],
        "malicious_accepted": [
            client_id for client_id in accepted_ids if client_id in malicious_ids
        ],
        "available_clients": current_app.extensions.get("available_clients", {}).get(
            round_id, []
        ),
        "attack_enabled": attack,
        "defense": payload.get("defense", "none"),
        "accuracy": round(accuracy, 4),
        "loss": round(loss, 4),
        "aggregated_delta": round(aggregated_delta, 4),
        "global_model_version": current_app.extensions["global_model"].get(
            "model_version", round_id
        ),
        "fedavg_samples": fedavg_result["total_samples"],
        "fedavg_client_weights": fedavg_result["client_weights"],
        "reason": "model_updated",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    current_app.extensions["round_results"].append(result)
    return jsonify(result)
