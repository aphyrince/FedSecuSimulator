from flask import Blueprint, current_app, jsonify, request

from clients.services.training_service import build_round_update, run_local_training
from clients.services.server_client import post_json
from clients.services.dlg_service import add_gaussian_noise, compute_single_image_gradient
from clients.services.dlg_data_loader import load_image, load_label
from pathlib import Path
from clients.services.collusion import collude_updates
from common.model import MODEL_FORMAT

experiment_bp = Blueprint("client_experiment", __name__)
PROJECT_ROOT_PATH = Path(__file__).resolve().parents[2]


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
    """중앙 서버가 전달한 전역 모델을 수신해 로컬 최신 모델로 저장한다."""
    payload = request.get_json(silent=True) or {}
    state = current_app.extensions["client_state"]
    if not state.training_lock.acquire(blocking=False):
        return jsonify(error="Client is already training"), 409
    try:
        # 실패한 학습 결과가 제출되지 않도록 시작 전에 이전 완료 상태를 초기화한다.
        state.latest_local_training = {}
        state.expired_rounds.discard(int(payload.get("round", 1)))
        state.receive_global_model(payload)
        training_record = run_local_training(state, current_app.config["CLIENT_ID"], payload)
    except TimeoutError as error:
        return jsonify(error=str(error)), 408
    except (ValueError, KeyError, TypeError, OSError) as error:
        return jsonify(error=str(error)), 400
    finally:
        state.training_lock.release()

    return jsonify(
        status="received",
        client_id=current_app.config["CLIENT_ID"],
        model_version=payload.get("model_version"),
        training={key: training_record[key] for key in ("round", "client_id", "status", "num_samples", "training_loss")},
    )


@experiment_bp.get("/api/model/global")
def get_global_model():
    return jsonify(current_app.extensions["client_state"].global_model)


@experiment_bp.post("/api/round/execute")
def execute_round():
    """실험 관리자가 요청한 로컬 라운드 업데이트를 생성한다."""
    payload = request.get_json(silent=True) or {}
    try:
        update = build_round_update(current_app.extensions["client_state"], current_app.config["CLIENT_ID"], payload)
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
        update = build_round_update(state, current_app.config["CLIENT_ID"], payload)
    except (ValueError, KeyError) as error:
        return jsonify(error=str(error)), 409
    if isinstance(payload.get("colluded_update"), dict):
        update = payload["colluded_update"]
    if update.get("client_id") != current_app.config["CLIENT_ID"] or update.get("round") != round_id:
        return jsonify(error="Submission identity mismatch"), 400
    server_url = payload.get("server_url", current_app.config["SERVER_URL"])
    response = post_json(
        f"{server_url.rstrip('/')}/api/rounds/{round_id}/updates",
        update,
    )
    state.save_submission(round_id, response)
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
        if own_update.get("client_id") != current_app.config["CLIENT_ID"] or own_update.get("round") != round_id:
            raise ValueError("Collusion identity mismatch")
        return jsonify(collude_updates(state.global_model, own_update, peer_updates,
                                        payload.get("attack_scale", -5.0), payload.get("min_colluders", 2)))
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
    state.set_malicious_peers(round_id, peers)
    return jsonify(
        status="received",
        client_id=current_app.config["CLIENT_ID"],
        round=round_id,
        peers=peers,
    )


@experiment_bp.post("/api/dlg/start")
def start_dlg():
    """원본 이미지와 체크포인트로 단일 이미지 그래디언트를 계산한다."""
    payload = request.get_json(silent=True) or {}
    if payload.get("global_model", {}).get("architecture") == MODEL_FORMAT:
        return jsonify(error="The role-2 DLG implementation must be connected to common.model before using the real CNN"), 501
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
    sigma = float(payload.get("noise_sigma", 0.0)) if defense_enabled else 0.0
    if defense_enabled:
        gradient = add_gaussian_noise(
            gradient,
            sigma=sigma,
            seed=int(payload.get("seed", 42))
            + round_id
            + current_app.config["CLIENT_ID"],
        )
    image_shape = payload.get("image_shape")
    if not image_shape:
        image_shape = list(image.shape if hasattr(image, "shape") else [len(image)])
    state = current_app.extensions["client_state"]
    state.dlg_gradient = gradient
    state.dlg_round = round_id
    server_url = payload.get("server_url", current_app.config["SERVER_URL"])
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
    return jsonify(status="submitted", gradient=gradient, server_response=response)


@experiment_bp.post("/api/round/update-share")
def receive_shared_malicious_update():
    """다른 악성 클라이언트의 추출값을 API로 수신해 저장한다."""
    payload = request.get_json(silent=True) or {}
    round_id = int(payload.get("round", 1))
    sender_id = payload.get("sender_id")
    update = payload.get("update")
    if sender_id is None or not isinstance(update, dict):
        return jsonify(error="sender_id and update are required"), 400
    current_app.extensions["client_state"].receive_malicious_update(
        round_id=round_id,
        sender_id=int(sender_id),
        update=update,
    )
    return jsonify(
        status="received",
        client_id=current_app.config["CLIENT_ID"],
        sender_id=int(sender_id),
        round=round_id,
    )


@experiment_bp.get("/api/training/status")
def get_training_status():
    """현재 클라이언트의 최신 로컬 학습 결과를 반환한다."""
    return jsonify(
        client_id=current_app.config["CLIENT_ID"],
        training=current_app.extensions["client_state"].latest_local_training,
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
        update=current_app.extensions["client_state"].get_extracted_update(round_id),
    )


@experiment_bp.get("/api/updates/submitted/<int:round_id>")
def get_submitted_update(round_id):
    """중앙 서버에 제출한 라운드 업데이트 응답을 반환한다."""
    return jsonify(
        client_id=current_app.config["CLIENT_ID"],
        round=round_id,
        submission=current_app.extensions["client_state"].submitted_updates.get(round_id),
    )
