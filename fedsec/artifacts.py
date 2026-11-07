"""Versioned tensor envelopes for HTTP multipart and shared checkpoints."""

from dataclasses import asdict
import io
import json
from pathlib import Path
import torch
from .config import DataConfig, ModelConfig, integer
from .contracts import ClientUpdate, UPDATE_DEFINITION
from .model import build_model
from .tensors import clone_state, state_hash


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def save_checkpoint(path: Path, *, state, model_config: ModelConfig, data_config: DataConfig,
                    experiment_id: str, model_version: int) -> None:
    integer("model_version", model_version)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"format_version": 1, "experiment_id": experiment_id,
                "model_version": model_version, "model_config": asdict(model_config),
                "preprocessing": {"normalization_mean": list(data_config.normalization_mean),
                                  "normalization_std": list(data_config.normalization_std)},
                "model_hash": state_hash(state), "model_state": clone_state(state)}, path)


def load_checkpoint(path: str | Path) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("format_version") != 1:
        raise ValueError("지원하지 않는 체크포인트 형식입니다.")
    required = {"experiment_id", "model_version", "model_config", "preprocessing", "model_hash", "model_state"}
    if not required.issubset(payload):
        raise ValueError("체크포인트 필드가 누락되었습니다.")
    integer("model_version", payload["model_version"])
    if state_hash(payload["model_state"]) != payload["model_hash"]:
        raise ValueError("체크포인트 가중치 해시가 다릅니다.")
    config = ModelConfig(**payload["model_config"])
    build_model(config, payload["model_state"])
    preprocessing = payload["preprocessing"]
    DataConfig(normalization_mean=preprocessing["normalization_mean"],
               normalization_std=preprocessing["normalization_std"])
    return payload


def encode_update(update: ClientUpdate) -> bytes:
    stream = io.BytesIO()
    torch.save({"format_version": 1, **update.metadata(), "delta": clone_state(update.delta)}, stream)
    return stream.getvalue()


def decode_update(data: bytes) -> ClientUpdate:
    payload = torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("format_version") != 1 or payload.get("definition") != UPDATE_DEFINITION:
        raise ValueError("지원하지 않는 업데이트 형식/정의입니다.")
    names = ("experiment_id", "client_id", "round_id", "model_version", "num_samples", "base_model_hash", "delta")
    if not set(names).issubset(payload):
        raise ValueError("업데이트 필드가 누락되었습니다.")
    return ClientUpdate(**{name: payload[name] for name in names})


def save_update(path: Path, update: ClientUpdate) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encode_update(update))
