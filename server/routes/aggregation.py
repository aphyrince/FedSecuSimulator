from datetime import datetime, timezone
import json
import random
from urllib.request import Request, urlopen

from flask import Blueprint, current_app, jsonify, request


aggregation_bp = Blueprint("server_aggregation", __name__)


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
    updates = [update for update in payload.get("updates", []) if update.get("valid", True)]
    if not updates:
        return jsonify(error="At least one valid update is required"), 400

    values = [update["update"]["model_delta"] for update in updates]
    if payload.get("defense") == "coordinate_median":
        values.sort()
        aggregated_delta = values[len(values) // 2]
    else:
        aggregated_delta = sum(values) / len(values)

    attack = payload.get("attack_enabled", False)
    accuracy = max(0.05, min(0.99, 0.50 + round_id * 0.025 + aggregated_delta * 0.1))
    if attack:
        accuracy -= 0.08
    if payload.get("defense") == "norm_clipping":
        accuracy -= 0.01
    elif payload.get("defense") == "coordinate_median" and attack:
        accuracy += 0.04

    selected_ids = payload.get("selected_clients", [])
    accepted_ids = [update["client_id"] for update in updates]
    malicious_ids = payload.get("malicious_clients", [])
    result = {
        "experiment_id": payload.get("experiment_id", "unknown"),
        "round": round_id,
        "model_version": round_id,
        "selected_clients": payload.get("selected_clients", []),
        "accepted_clients": accepted_ids,
        "delayed_clients": payload.get("delay_clients", []),
        "malicious_clients": payload.get("malicious_clients", []),
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
        "loss": round(max(0.05, 1.0 - accuracy), 4),
        "aggregated_delta": round(aggregated_delta, 4),
        "reason": "model_updated",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    current_app.extensions["round_results"].append(result)
    return jsonify(result)


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

    seed = config.get("seed", 42)
    participation = config.get("participation_rate", 0.5)
    nums_clients = config.get("nums_clients", len(registered))
    selected_count = max(1, round(nums_clients * participation))
    rng = random.Random(seed + round_id)
    selected = sorted(rng.sample(range(len(available_clients)), min(selected_count, len(available_clients))))
    selected_clients = [available_clients[index] for index in selected]
    selected_ids = [client["client_id"] for client in selected_clients]
    current_app.extensions["selected_clients"][round_id] = selected_ids

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
                    f"{client['url']}/api/round/execute",
                    {
                        "round": round_id,
                        "malicious": client["client_id"] in malicious_ids,
                        "attack_enabled": config.get("attack_enabled", False),
                    },
                )
            )
        except (OSError, ValueError):
            current_app.logger.warning("Client update failed: client_id=%s", client["client_id"])

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

    values = [update["update"]["model_delta"] for update in updates]
    if payload.get("defense") == "coordinate_median":
        values.sort()
        aggregated_delta = values[len(values) // 2]
    else:
        aggregated_delta = sum(values) / len(values)

    attack = payload.get("attack_enabled", False)
    accuracy = max(0.05, min(0.99, 0.50 + round_id * 0.025 + aggregated_delta * 0.1))
    if attack:
        accuracy -= 0.08
    if payload.get("defense") == "norm_clipping":
        accuracy -= 0.01
    elif payload.get("defense") == "coordinate_median" and attack:
        accuracy += 0.04

    selected_ids = payload.get("selected_clients", [])
    accepted_ids = [update["client_id"] for update in updates]
    malicious_ids = payload.get("malicious_clients", [])
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
        "loss": round(max(0.05, 1.0 - accuracy), 4),
        "aggregated_delta": round(aggregated_delta, 4),
        "reason": "model_updated",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    current_app.extensions["round_results"].append(result)
    return jsonify(result)
