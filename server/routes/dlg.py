from flask import Blueprint, current_app, jsonify, request
import json
from urllib.error import URLError
from urllib.request import Request, urlopen

from server.services.dlg_attack import create_dummy_image, reconstruct_image


dlg_bp = Blueprint("server_dlg", __name__)


def _post_json(url, payload, timeout=180.0):
    request = Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


@dlg_bp.post("/api/dlg/gradient")
def receive_dlg_gradient():
    """클라이언트 그래디언트를 받아 DLG 복원을 수행한다."""
    payload = request.get_json(silent=True) or {}
    round_id = int(payload.get("round", 1))
    gradient = payload.get("gradient", {})
    config = current_app.extensions.setdefault("round_config", {})
    if gradient.get("format") == "small_cnn_gradient_v1":
        from security.dlg_api import reconstruct_gradient_payload
        global_model = current_app.extensions.get("global_model")
        if not global_model:
            return jsonify(error="Global CNN model is missing"), 409
        try:
            result = reconstruct_gradient_payload(
                global_model, gradient,
                steps=int(config.get("dlg_iterations", 2)),
                seed=int(config.get("seed", 42)) + round_id,
            )
        except (ValueError, TypeError, RuntimeError, MemoryError) as exc:
            return jsonify(error=f"Real DLG reconstruction failed: {exc}"), 400
        dlg_result = {
            "round": round_id,
            "client_id": payload.get("client_id"),
            "model_version": payload.get("model_version"),
            "noise_sigma": payload.get("noise_sigma", 0.0),
            "defense_enabled": payload.get("defense_enabled", False),
            **result,
        }
        current_app.extensions.setdefault("dlg_results", {})[round_id] = dlg_result
        callback_url = config.get("experiment_manager_url")
        if not callback_url:
            return jsonify(error="DLG manager callback URL missing"), 409
        # 평가 관리자가 원본 이미지와 MSE를 계산하며, 서버는 원본을 알 수 없다.
        _post_json(callback_url.rstrip("/") + "/api/dlg/results", dlg_result)
        return jsonify(status="completed", round=round_id,
                       gradient_loss=dlg_result["gradient_loss"],
                       predicted_label=dlg_result["predicted_label"])
    global_model = current_app.extensions.get("global_model") or {
        "weights": [1.0] * max(1, int(__import__("math").prod(payload.get("image_shape", [1]))))
    }
    image_shape = payload.get("image_shape", [len(gradient.get("values", []))])
    participant = current_app.extensions.setdefault("dlg_participants", {}).get(
        round_id, {}
    )
    dummy_image = participant.get("dummy_image") or create_dummy_image(
        global_model=global_model,
        image_shape=image_shape,
        seed=int(config.get("seed", 42)) + round_id,
    )
    participant["dummy_image"] = dummy_image
    result = reconstruct_image(
        global_model=global_model,
        target_gradient=gradient,
        image_shape=image_shape,
        label=int(payload.get("label", 0)),
        seed=int(config.get("seed", 42)) + round_id,
        iterations=int(config.get("dlg_iterations", 200)),
        dummy_image=dummy_image,
    )
    current_app.extensions.setdefault("dlg_results", {})[round_id] = {
        "round": round_id,
        "client_id": payload.get("client_id"),
        "model_version": payload.get("model_version"),
        "noise_sigma": payload.get("noise_sigma", 0.0),
        "defense_enabled": payload.get("defense_enabled", False),
        "initial_dummy_image": dummy_image,
        **result,
    }
    dlg_result = current_app.extensions["dlg_results"][round_id]
    callback_url = config.get("experiment_manager_url")
    if callback_url:
        try:
            _post_json(callback_url.rstrip("/") + "/api/dlg/results", dlg_result)
        except (OSError, ValueError, URLError):
            current_app.logger.warning(
                "DLG result callback failed: round=%s", round_id
            )
    return jsonify(status="completed", dlg=current_app.extensions["dlg_results"][round_id])


@dlg_bp.post("/api/dlg/participant")
def start_dlg_participant():
    """DLG 참가 클라이언트 정보를 받아 저장된 체크포인트를 전달한다."""
    payload = request.get_json(silent=True) or {}
    client = payload.get("client", {})
    client_url = client.get("url")
    client_id = int(client.get("client_id", -1))
    if client_id < 0 or not client_url:
        return jsonify(error="Valid DLG participant client information is required"), 400

    round_id = int(payload.get("round", 1))
    checkpoint = current_app.extensions.get("global_model")
    if checkpoint is None:
        return jsonify(error="Trained global model checkpoint is not available"), 409

    current_app.extensions.setdefault("dlg_participants", {})[round_id] = {
        **client,
        "round": round_id,
        "dummy_image": create_dummy_image(
            global_model=checkpoint,
            image_shape=payload.get("image_shape", [1]),
            seed=int(payload.get("seed", 42)) + round_id,
        ),
    }
    try:
        result = _post_json(
            f"{client_url.rstrip('/')}/api/dlg/start",
            {
                **payload,
                "global_model": checkpoint,
                "model_version": checkpoint.get("model_version", 0),
                "server_url": payload.get("server_url", request.host_url.rstrip("/")),
            },
        )
    except (OSError, ValueError, URLError) as exc:
        return jsonify(
            error="Failed to send checkpoint to DLG participant",
            detail=str(exc),
        ), 502
    return jsonify(
        status="checkpoint_sent",
        round=round_id,
        client=result.get("client_id", client_id),
        client_url=client_url,
        model_version=checkpoint.get("model_version", 0),
    )
