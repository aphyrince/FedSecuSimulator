"""Tensor 체크포인트의 가중치와 모델 복원 가능성을 검증한다."""

from pathlib import Path

import torch

from fedsec.model import PREPROCESSING, SUPPORTED_ARCHITECTURES, create_model, model_config
from fedsec.tensors import copy_state, state_hash


def load_checkpoint(path: str | Path) -> dict:
    payload = torch.load(path, weights_only=True, map_location="cpu")
    if payload.get("format_version") != 1 or state_hash(payload["weights"]) != payload.get("model_hash"):
        raise ValueError("Invalid model checkpoint")
    architecture = payload.get("architecture")
    if architecture not in SUPPORTED_ARCHITECTURES or payload.get("preprocessing") != PREPROCESSING:
        raise ValueError("Unsupported checkpoint architecture or preprocessing")
    config = model_config(payload.get("model_config"), architecture=architecture)
    model = create_model(config=config)
    weights = copy_state(payload["weights"], model.state_dict())
    model.load_state_dict(weights, strict=True)
    return payload
