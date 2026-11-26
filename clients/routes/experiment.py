
from pathlib import Path
import threading
import time
from datetime import datetime, timezone

from flask import Blueprint, current_app, jsonify, request

from clients.services.training_service import build_round_update, run_local_training
from clients.services.server_client import post_json, post_tensor_update
from clients.services.dlg_service import add_gaussian_noise, compute_single_image_gradient
from clients.services.dlg_data_loader import load_image, load_label
from clients.services.collusion import collude_updates
from common.model import MODEL_FORMAT, decode_wire_weights, encode_weights, encode_wire_weights, finite_number
from common.runtime_log import runtime_logger
from fedsec.models.resnet18 import RESNET18_FORMAT


experiment_bp = Blueprint("client_experiment", __name__)
PROJECT_ROOT_PATH = Path(__file__).resolve().parents[2]


def _submit_round_update(
    state, client_id: int, payload: dict, default_server_url: str
) -> dict:
    round_id = int(payload.get("round", 1))
    runtime_logger.info(
        "client preparing update experiment=%s round=%s client=%s",
        state.global_model.get("experiment_id"), round_id, client_id,
    )
    attacked = payload.get("malicious") and payload.get("attack_enabled")
    tensor_update = state.global_model.get("architecture") == RESNET18_FORMAT and not attacked
    if tensor_update:
        update = build_round_update(state, client_id, payload, tensor_payload=True)
    else:
        update = build_round_update(state, client_id, payload)

    if attacked:
        if "malicious_peers" in payload:
            peers = payload["malicious_peers"]
        elif round_id in state.malicious_peers:
            peers = state.malicious_peers[round_id]
        else:
            raise ValueError("Malicious peer information is missing")
        if not isinstance(peers, list):
            raise ValueError("Malicious peer information must be a list")

        peer_updates = []
        for peer in peers:
            peer_id = peer.get("client_id")
            peer_url = peer.get("url")
            if (
                type(peer_id) is not int
                or peer_id == client_id
                or not isinstance(peer_url, str)
                or not peer_url
            ):
                raise ValueError("Invalid malicious peer information")
            peer_url = peer_url.rstrip("/")
            exchange_payload = {
                "round": round_id,
                "sender_id": client_id,
                "update": update,
            }
            remaining = _round_time_remaining(payload)
            if remaining <= 0:
                raise TimeoutError(
                    f"Timed out waiting for malicious peer {peer_id}"
                )
            exchange = post_json(
                f"{peer_url}/api/round/update-share",
                exchange_payload,
                timeout=min(60.0, remaining),
                compact=True,
            )
            while exchange.get("status") == "pending":
                remaining = _round_time_remaining(payload)
                if remaining <= 0:
                    raise TimeoutError(
                        f"Timed out waiting for malicious peer {peer_id}"
                    )
                time.sleep(min(0.1, remaining))
                exchange = post_json(
                    f"{peer_url}/api/round/update-share",
                    {
                        "round": round_id,
                        "sender_id": client_id,
                        "poll": True,
                    },
                    timeout=min(60.0, remaining),
                    compact=True,
                )
            if exchange.get("status") == "expired":
                raise TimeoutError(
                    f"Malicious peer {peer_id} expired before sharing its update"
                )
            peer_update = exchange.get("peer_update")
            if (
                not isinstance(peer_update, dict)
                or peer_update.get("client_id") != peer_id
                or peer_update.get("round") != round_id
            ):
                raise ValueError("Malicious peer update identity mismatch")
            peer_updates.append(peer_update)

        update = collude_updates(
            state.global_model,
            update,
            peer_updates,
            payload.get(
                "attack_scale",
                state.attack_scale_by_round.get(round_id, -5.0),
            ),
        )

    if update.get("client_id") != client_id or update.get("round") != round_id:
        raise ValueError("Submission identity mismatch")

    server_url = payload.get("server_url", default_server_url)
    url = f"{server_url.rstrip('/')}/api/rounds/{round_id}/updates"
    if tensor_update:
        response = post_tensor_update(url, update, timeout=60.0)
    else:
        local_model = update.get("local_model", {})
        if local_model.get("architecture") == RESNET18_FORMAT:
            update["local_model"] = {**local_model, "weights": encode_wire_weights(local_model["weights"])}
            update["update"] = {**update["update"], "values": encode_wire_weights(update["update"]["values"])}
            update["tensor_encoding"] = "base64_float32"
        response = post_json(url, update, timeout=60.0, compact=True)
    state.save_submission(round_id, response)
    return response


def _round_time_remaining(payload: dict) -> float:
    deadline_value = payload.get("update_deadline")
    if deadline_value is None:
        return float(payload.get("round_timeout", 60.0))
    if not isinstance(deadline_value, str):
        raise ValueError("update_deadline must be an ISO timestamp")
    try:
        deadline = datetime.fromisoformat(deadline_value)
    except ValueError as error:
        raise ValueError("update_deadline must be an ISO timestamp") from error
    if deadline.tzinfo is None:
        deadline = deadline.replace(tzinfo=timezone.utc)
    return max(0.0, (deadline - datetime.now(timezone.utc)).total_seconds())


def _report_training_failure(app, payload: dict, client_id: int, error: Exception):
    round_id = int(payload.get("round", 1))
    state = app.extensions["client_state"]
    status = "expired" if isinstance(error, TimeoutError) else "failed"
    runtime_logger.exception(
        "client training or submission failed experiment=%s round=%s client=%s status=%s",
        payload.get("experiment_id"), round_id, client_id, status,
    )
    state.fail_training(round_id, str(error), status=status)
    try:
        post_json(
            f"{payload['server_url'].rstrip('/')}/api/rounds/{round_id}/failures",
            {
                "round": round_id,
                "client_id": client_id,
                "reason": "training_timeout" if status == "expired" else "client_failure",
                "error": str(error),
            },
            timeout=3.0,
        )
    except (OSError, ValueError):
        app.logger.exception(
            "Failed to report client round failure: round=%s client_id=%s",
            round_id,
            client_id,
        )


def _train_and_submit(app, payload: dict):
    state = app.extensions["client_state"]
    client_id = app.config["CLIENT_ID"]
    round_id = int(payload.get("round", 1))
    try:
        with app.app_context():
            round_context = payload.get("round_context", {})
            submission_payload = {
                "round": round_id,
                "malicious": bool(round_context.get("malicious", False)),
                "attack_enabled": bool(payload.get("attack_enabled", False)),
                "attack_scale": round_context.get(
                    "attack_scale",
                    state.attack_scale_by_round.get(round_id, -5.0),
                ),
                "update_deadline": payload.get("update_deadline"),
                "round_timeout": payload.get("training_config", {}).get(
                    "round_timeout", 60.0
                ),
                "server_url": payload["server_url"],
            }
            if "malicious_peers" in round_context:
                submission_payload["malicious_peers"] = round_context[
                    "malicious_peers"
                ]
            if payload.get("architecture") == RESNET18_FORMAT and not (
                submission_payload["malicious"] and submission_payload["attack_enabled"]
            ):
                run_local_training(state, client_id, payload, tensor_payload=True)
            else:
                run_local_training(state, client_id, payload)
            _submit_round_update(
                state,
                client_id,
                submission_payload,
                app.config["SERVER_URL"],
            )
    except (
        OSError,
        ValueError,
        KeyError,
        TypeError,
        RuntimeError,
        TimeoutError,
    ) as error:
        with app.app_context():
            _report_training_failure(app, payload, client_id, error)
    finally:
        state.training_lock.release()


@experiment_bp.post("/api/experiment/config")
def set_config():
    # 실험에서 사용할 라운드·참여·공격·방어 설정을 저장한다.
    app = current_app._get_current_object()
    state = app.extensions["client_state"]
    config = state.set_experiment_config(request.get_json(silent=True) or {})
    return jsonify(
        status="configured",
        client_id=app.config["CLIENT_ID"],
        config=config,
    )


@experiment_bp.get("/api/experiment/config")
def get_config():
    return jsonify(current_app.extensions["client_state"].experiment_config)


@experiment_bp.post("/api/model/global")
def receive_global_model():
    """전역 모델을 저장하고 백그라운드 학습을 시작한다."""
    payload = request.get_json(silent=True) or {}
    state = current_app.extensions["client_state"]
    if not isinstance(payload, dict):
        return jsonify(error="Global model payload must be an object"), 400
    try:
        round_id = int(payload.get("round", 1))
        server_url = payload.get("server_url", current_app.config["SERVER_URL"])
        if not isinstance(server_url, str) or not server_url:
            raise ValueError("server_url is required")
        payload["server_url"] = server_url.rstrip("/")
    except (TypeError, ValueError) as error:
        return jsonify(error=str(error)), 400

    if not state.training_lock.acquire(blocking=False):
        return jsonify(error="Client is already training"), 409

    try:
        runtime_logger.info(
            "client received global model experiment=%s round=%s client=%s bytes=%s format=%s",
            payload.get("experiment_id"), round_id, current_app.config["CLIENT_ID"],
            request.content_length, payload.get("tensor_encoding", "numeric JSON"),
        )
        if "tensor_encoding" in payload:
            if payload.pop("tensor_encoding") != "base64_float32":
                raise ValueError("Unsupported global model tensor encoding")
            payload["weights"] = decode_wire_weights(payload["weights"])
        state.latest_local_training = {}
        state.expired_rounds.discard(round_id)
        state.receive_global_model(payload)
        round_context = payload.get("round_context", {})
        if not isinstance(round_context, dict):
            raise ValueError("round_context must be an object")
        if "malicious_peers" in round_context:
            malicious_peers = round_context["malicious_peers"]
            if not isinstance(malicious_peers, list):
                raise ValueError("malicious_peers must be a list")
            state.set_malicious_peers(
                round_id,
                malicious_peers,
                round_context.get(
                    "attack_scale", state.attack_scale_by_round.get(round_id, -5.0)
                ),
            )
        state.begin_training(
            round_id,
            0,
            int(payload.get("training_config", {}).get("local_epochs", 1)),
            malicious=bool(round_context.get("malicious", False)),
            delayed=bool(round_context.get("delayed", False)),
        )
        worker = threading.Thread(
            target=_train_and_submit,
            args=(current_app._get_current_object(), payload),
            name=f"client-{current_app.config['CLIENT_ID']}-round-{round_id}",
            daemon=True,
        )
        worker.start()
    except (ValueError, KeyError, TypeError, OSError, RuntimeError) as error:
        state.fail_training(round_id, str(error))
        state.training_lock.release()
        return jsonify(error=str(error)), 400

    return jsonify(
        status="training",
        client_id=current_app.config["CLIENT_ID"],
        round=round_id,
        model_version=payload.get("model_version"),
    ), 202


@experiment_bp.get("/api/model/global")
def get_global_model():
    model = current_app.extensions["client_state"].global_model
    if model.get("architecture") == RESNET18_FORMAT:
        model = {**model, "weights": encode_weights(model["weights"])}
    return jsonify(model)


@experiment_bp.post("/api/round/execute")
def execute_round():
    """실험 관리자가 요청한 로컬 라운드 업데이트를 생성한다."""
    payload = request.get_json(silent=True) or {}

    try:
        update = build_round_update(
            current_app.extensions["client_state"],
            current_app.config["CLIENT_ID"],
            payload,
        )
    except (ValueError, KeyError) as error:
        return jsonify(error=str(error)), 409

    return jsonify(update)


@experiment_bp.post("/api/round/submit")
def submit_round_update():
    """추출한 로컬 업데이트를 중앙 서버에 직접 제출한다."""
    payload = request.get_json(silent=True) or {}
    round_id = int(payload.get("round", 1))
    state = current_app.extensions["client_state"]

    try:
        response = _submit_round_update(
            state,
            current_app.config["CLIENT_ID"],
            payload,
            current_app.config["SERVER_URL"],
        )
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
        return jsonify(error=str(error)), 400

    return jsonify(
        status="submitted",
        client_id=current_app.config["CLIENT_ID"],
        round=round_id,
        server_response=response,
    )


@experiment_bp.post("/api/round/collusion")
def collude_round_updates():
    """악성 클라이언트끼리 업데이트를 평균한 뒤 공격 배율을 적용한다."""
    payload = request.get_json(silent=True) or {}
    round_id = int(payload.get("round", 1))
    peer_updates = payload.get("peer_updates", [])
    own_update = payload.get("own_update", {})
    state = current_app.extensions["client_state"]

    if not peer_updates:
        peer_updates = list(
            state.received_malicious_updates.get(round_id, {}).values()
        )

    try:
        if (
            own_update.get("client_id") != current_app.config["CLIENT_ID"]
            or own_update.get("round") != round_id
        ):
            raise ValueError("Collusion identity mismatch")

        attack_scale = state.attack_scale_by_round.get(
            round_id,
            payload.get("attack_scale", -5.0),
        )

        return jsonify(
            collude_updates(
                state.global_model,
                own_update,
                peer_updates,
                attack_scale,
                payload.get("min_colluders", 2),
            )
        )
    except (ValueError, TypeError, KeyError) as error:
        return jsonify(error=str(error)), 400


@experiment_bp.post("/api/round/malicious-peers")
def receive_malicious_peers():
    """실험 관리자가 전달한 다른 악성 클라이언트 정보를 저장한다."""
    payload = request.get_json(silent=True) or {}
    round_id = int(payload.get("round", 1))
    peers = payload.get("peers", [])

    if not isinstance(peers, list):
        return jsonify(error="peers must be a list"), 400

    state = current_app.extensions["client_state"]

    try:
        attack_scale = finite_number(
            payload.get("attack_scale", -5.0),
            "attack_scale",
        )
        state.set_malicious_peers(round_id, peers, attack_scale)
    except (ValueError, TypeError) as error:
        return jsonify(error=str(error)), 400

    return jsonify(
        status="received",
        client_id=current_app.config["CLIENT_ID"],
        round=round_id,
        peers=peers,
        attack_scale=attack_scale,
    )


@experiment_bp.post("/api/dlg/start")
def start_dlg():
    """원본 이미지와 체크포인트로 단일 이미지 그래디언트를 계산한다."""
    payload = request.get_json(silent=True) or {}

    if payload.get("global_model", {}).get("architecture") == MODEL_FORMAT:
        from fedsec.data import CifarRangeDataset
        from fedsec.model import PREPROCESSING
        from security.dlg_api import make_gradient_payload, original_pixels

        state = current_app.extensions["client_state"]

        try:
            if "dlg_image" in payload:
                import math
                import torch

                pixels = payload["dlg_image"]
                label = payload.get("dlg_label")

                if (
                    not isinstance(pixels, list)
                    or len(pixels) != 3 * 32 * 32
                    or any(
                        isinstance(value, bool)
                        or not isinstance(value, (int, float))
                        or not math.isfinite(value)
                        or not 0 <= value <= 1
                        for value in pixels
                    )
                ):
                    raise ValueError(
                        "DLG image must contain 3072 pixels in [0,1]"
                    )

                if type(label) is not int or not 0 <= label <= 9:
                    raise ValueError(
                        "DLG label must be an integer from 0 to 9"
                    )

                image = torch.tensor(
                    pixels,
                    dtype=torch.float32,
                ).reshape(3, 32, 32)

                mean = torch.tensor(
                    PREPROCESSING["mean"]
                ).reshape(3, 1, 1)

                std = torch.tensor(
                    PREPROCESSING["std"]
                ).reshape(3, 1, 1)

                sample = (image - mean) / std

            else:
                sample, label = CifarRangeDataset(
                    state.dataset_range,
                    limit=1,
                )[0]

            sigma = (
                float(payload.get("noise_sigma", 0.0))
                if payload.get("defense_enabled")
                else 0.0
            )

            gradient = make_gradient_payload(
                payload["global_model"],
                sample,
                label,
                sigma=sigma,
                seed=(
                    int(payload.get("seed", 42))
                    + int(payload.get("round", 1))
                    + current_app.config["CLIENT_ID"]
                ),
            )

        except (ValueError, TypeError, OSError, RuntimeError) as exc:
            return jsonify(
                error=f"CIFAR-10 DLG gradient failed: {exc}"
            ), 400

        manager_url = payload.get("experiment_manager_url")

        if not manager_url:
            return jsonify(
                error="experiment_manager_url is required for DLG evaluation"
            ), 400

        round_id = int(payload.get("round", 1))
        client_id = current_app.config["CLIENT_ID"]

        # 원본 이미지는 평가 관리자에게만 전달한다.
        post_json(
            f"{manager_url.rstrip('/')}/api/dlg/original",
            {
                "experiment_id": payload["global_model"]["experiment_id"],
                "round": round_id,
                "client_id": client_id,
                "image": original_pixels(sample),
                "label": label,
            },
            timeout=30.0,
        )

        state.dlg_gradient = {
            "format": gradient["format"],
            "noise_sigma": sigma,
        }
        state.dlg_round = round_id

        server_url = payload.get(
            "server_url",
            current_app.config["SERVER_URL"],
        )

        response = post_json(
            f"{server_url.rstrip('/')}/api/dlg/gradient",
            {
                "round": round_id,
                "client_id": client_id,
                "model_version": payload["global_model"].get(
                    "model_version", 0
                ),
                "noise_sigma": sigma,
                "defense_enabled": bool(
                    payload.get("defense_enabled", False)
                ),
                "image_shape": [3, 32, 32],
                "gradient": gradient,
            },
            timeout=float(payload.get("round_timeout", 900)) + 120.0,
        )

        return jsonify(
            status="submitted",
            round=round_id,
            client_id=client_id,
            server_status=response.get("status"),
        )

    round_id = int(payload.get("round", 1))
    image_path = payload.get("image_path", "")

    image = (
        load_image(image_path, PROJECT_ROOT_PATH)
        if image_path
        else payload.get("image", [])
    )

    label = load_label(
        payload.get("label_path", ""),
        payload.get("label", 0),
        PROJECT_ROOT_PATH,
    )

    model = payload.get("global_model", {})
    gradient = compute_single_image_gradient(model, image, label)

    defense_enabled = bool(payload.get("defense_enabled", False))
    sigma = (
        float(payload.get("noise_sigma", 0.0))
        if defense_enabled
        else 0.0
    )

    if defense_enabled:
        gradient = add_gaussian_noise(
            gradient,
            sigma=sigma,
            seed=(
                int(payload.get("seed", 42))
                + round_id
                + current_app.config["CLIENT_ID"]
            ),
        )

    image_shape = payload.get("image_shape")

    if not image_shape:
        image_shape = list(
            image.shape
            if hasattr(image, "shape")
            else [len(image)]
        )

    state = current_app.extensions["client_state"]
    state.dlg_gradient = gradient
    state.dlg_round = round_id

    server_url = payload.get(
        "server_url",
        current_app.config["SERVER_URL"],
    )

    response = post_json(
        f"{server_url.rstrip('/')}/api/dlg/gradient",
        {
            "round": round_id,
            "client_id": current_app.config["CLIENT_ID"],
            "model_version": payload.get("model_version", 0),
            "noise_sigma": sigma,
            "defense_enabled": defense_enabled,
            "label": label,
            "image_shape": image_shape,
            "gradient": gradient,
        },
    )

    return jsonify(
        status="submitted",
        gradient=gradient,
        server_response=response,
    )


@experiment_bp.post("/api/round/update-share")
def receive_shared_malicious_update():
    """다른 악성 클라이언트의 추출값을 API로 수신해 저장한다."""
    payload = request.get_json(silent=True) or {}
    round_id = int(payload.get("round", 1))
    sender_id = payload.get("sender_id")
    update = payload.get("update")

    if type(sender_id) is not int:
        return jsonify(error="sender_id is required"), 400

    state = current_app.extensions["client_state"]
    allowed_peers = {
        peer.get("client_id")
        for peer in state.malicious_peers.get(round_id, [])
    }
    if sender_id not in allowed_peers:
        return jsonify(error="Sender is not a configured malicious peer"), 403

    if payload.get("poll") is True and "update" in payload:
        return jsonify(error="Update polls must not include an update"), 400
    if payload.get("poll") is not True and (
        not isinstance(update, dict)
        or update.get("client_id") != sender_id
        or update.get("round") != round_id
    ):
        return jsonify(error="Malicious peer update identity mismatch"), 400
    if payload.get("poll") is not True:
        state.receive_malicious_update(
            round_id=round_id,
            sender_id=sender_id,
            update=update,
        )
    if round_id in state.expired_rounds:
        return jsonify(status="expired", peer_update=None)
    if state.latest_local_training.get("round") != round_id:
        return jsonify(
            status="pending",
            client_id=current_app.config["CLIENT_ID"],
            sender_id=sender_id,
            round=round_id,
            peer_update=None,
        ), 202
    try:
        peer_update = build_round_update(
            state,
            current_app.config["CLIENT_ID"],
            {
                "round": round_id,
                "malicious": True,
                "attack_enabled": True,
            },
        )
    except (ValueError, KeyError) as error:
        return jsonify(error=str(error)), 409

    return jsonify(
        status="ready",
        client_id=current_app.config["CLIENT_ID"],
        sender_id=sender_id,
        round=round_id,
        peer_update=peer_update,
    )


@experiment_bp.get("/api/training/status")
def get_training_status():
    """현재 클라이언트의 최신 로컬 학습 결과를 반환한다."""
    training = current_app.extensions["client_state"].latest_local_training
    local_model = training.get("local_model", {})
    if local_model.get("architecture") == RESNET18_FORMAT:
        delta = encode_weights(training["model_delta"])
        training = {
            **training,
            "local_model": {**local_model, "weights": encode_weights(local_model["weights"])},
            "model_delta": delta,
            "extracted_update": {**training["extracted_update"], "values": delta},
        }
    return jsonify(
        client_id=current_app.config["CLIENT_ID"],
        training=training,
    )


@experiment_bp.get("/api/training/history")
def get_training_history():
    """라운드별 로컬 학습 결과 기록을 반환한다."""
    state = current_app.extensions["client_state"]

    return jsonify(
        client_id=current_app.config["CLIENT_ID"],
        count=len(state.local_training_results),
        results=state.local_training_results,
    )


@experiment_bp.get("/api/updates/extracted/<int:round_id>")
def get_extracted_update(round_id):
    """학습 전후 모델 차이로 추출한 라운드 업데이트를 반환한다."""
    return jsonify(
        client_id=current_app.config["CLIENT_ID"],
        round=round_id,
        update=current_app.extensions[
            "client_state"
        ].get_extracted_update(round_id),
    )


@experiment_bp.get("/api/updates/submitted/<int:round_id>")
def get_submitted_update(round_id):
    """중앙 서버에 제출한 라운드 업데이트 응답을 반환한다."""
    return jsonify(
        client_id=current_app.config["CLIENT_ID"],
        round=round_id,
        submission=current_app.extensions[
            "client_state"
        ].submitted_updates.get(round_id),
    )
