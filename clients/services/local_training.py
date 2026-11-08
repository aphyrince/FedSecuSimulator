"""REST 연결부: 모델·데이터 요청을 변환하고 fedsec 호출 결과를 응답 형식으로 바꾼다."""

from copy import deepcopy
from datetime import datetime, timezone
from common.data import CifarRangeDataset
from common.model import build_model, decode_weights, encode_weights, weights_hash, weights_norm
from common.training import TrainingSettings
from fedsec.training import train_local_model as train_core
from fedsec.tensors import extract_update


def count_dataset_samples(dataset_range: dict) -> int:
    return sum(segment["end"] - segment["start"] for segment in dataset_range.get("ranges", []))


def train_local_model(global_model: dict, dataset_range: dict, training_config: dict,
                      round_id: int, client_id: int) -> dict:
    settings = TrainingSettings.from_dict(training_config)
    dataset = CifarRangeDataset(dataset_range, settings.max_train_samples)
    deadline = global_model.get("update_deadline")
    deadline = datetime.fromisoformat(deadline) if deadline else None

    def check_deadline():
        if deadline and datetime.now(timezone.utc) >= deadline:
            raise TimeoutError("Local training exceeded the round deadline")

    result = train_core(build_model(global_model), dataset, settings,
                        seed=training_config.get("seed", 42), round_id=round_id, client_id=client_id,
                        check_cancelled=check_deadline)
    local = {key: deepcopy(value) for key, value in global_model.items()
             if key not in ("weights", "training_config", "update_deadline")}
    local.update(weights=encode_weights(result.weights), model_version=int(global_model["model_version"]) + 1)
    return {"local_model": local, "model_delta": encode_weights(result.delta), "num_samples": result.num_samples,
            "training_loss": result.history[-1]["loss"], "training_accuracy": result.history[-1]["accuracy"],
            "history": result.history, "dataset_fingerprint": dataset.fingerprint,
            "training_config": settings.to_dict(), "base_model_hash": weights_hash(global_model["weights"]),
            "base_model_version": global_model["model_version"]}


def extract_model_update(global_model_before_training: dict, local_model_after_training: dict) -> dict:
    delta = extract_update(decode_weights(global_model_before_training["weights"]),
                           decode_weights(local_model_after_training["weights"]))
    return {"values": encode_weights(delta), "update_norm": weights_norm(delta)}
