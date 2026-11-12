import csv
import io
import json
import math
import re

import torch

from common.model import build_model, create_model, decode_weights, encode_weights, weights_hash
from fedsec.model import MODEL_FORMAT, PREPROCESSING, model_config


def parse_pretrained_global_model(
    raw_data: str, data_format: str, default_config: dict
) -> dict:
    """Parse JSON checkpoints or flat CSV/text values into the supported model payload."""
    if not isinstance(raw_data, str) or not raw_data.strip():
        raise ValueError("전역 모델 파일을 선택하거나 모델 값을 입력하세요.")
    if len(raw_data.encode("utf-8")) > 32 * 1024 * 1024:
        raise ValueError("전역 모델 입력은 32 MiB 이하여야 합니다.")

    if data_format == "json":
        try:
            payload = json.loads(raw_data)
        except json.JSONDecodeError as error:
            raise ValueError(f"전역 모델 JSON 형식이 올바르지 않습니다: {error.msg}") from error
        if not isinstance(payload, dict):
            raise ValueError("JSON 전역 모델은 객체여야 합니다.")
        if isinstance(payload.get("global_model"), dict):
            payload = payload["global_model"]
        if "state_dict" in payload and "weights" not in payload:
            payload = {**payload, "weights": payload["state_dict"]}
        checkpoint_config = model_config(payload.get("model_config", default_config))
        template = create_model(seed=0, config=checkpoint_config)
        raw_weights = payload.get("weights", payload.get("state_dict"))
        if raw_weights is None:
            raw_weights = {
                name: value
                for name, value in payload.items()
                if name in template["weights"]
            }
        if isinstance(raw_weights, list):
            raw_weights = _reshape_flat_values(
                [float(value) for value in raw_weights], template["weights"]
            )
        if not isinstance(raw_weights, dict):
            raise ValueError("JSON에는 weights 객체 또는 state_dict가 필요합니다.")
        if raw_weights and all(key.startswith("module.") for key in raw_weights):
            raw_weights = {
                key.removeprefix("module."): value
                for key, value in raw_weights.items()
            }
    elif data_format in {"csv", "text"}:
        checkpoint_config = model_config(default_config)
        template = create_model(seed=0, config=checkpoint_config)
        values = _parse_flat_values(raw_data, data_format)
        raw_weights = _reshape_flat_values(values, template["weights"])
    else:
        raise ValueError("지원하지 않는 전역 모델 입력 형식입니다.")

    if (
        data_format == "json"
        and payload.get("architecture", MODEL_FORMAT) != MODEL_FORMAT
    ):
        raise ValueError(f"지원하는 모델 구조는 {MODEL_FORMAT}입니다.")

    weights = encode_weights(
        decode_weights(raw_weights, decode_weights(template["weights"]))
    )
    checkpoint = {
        **template,
        "architecture": MODEL_FORMAT,
        "model_config": checkpoint_config,
        "preprocessing": payload.get("preprocessing", PREPROCESSING)
        if data_format == "json"
        else PREPROCESSING,
        "model_version": int(payload.get("model_version", 0))
        if data_format == "json"
        else 0,
        "weights": weights,
    }
    if checkpoint["model_version"] < 0:
        raise ValueError("model_version은 0 이상이어야 합니다.")
    build_model(checkpoint)
    checkpoint["initial_model_hash"] = weights_hash(weights)
    return checkpoint


def _parse_flat_values(raw_data: str, data_format: str) -> list[float]:
    tokens = []
    if data_format == "csv":
        try:
            rows = csv.reader(io.StringIO(raw_data))
            for row in rows:
                tokens.extend(cell.strip() for cell in row if cell.strip())
        except csv.Error as error:
            raise ValueError("CSV 전역 모델 파일 형식이 올바르지 않습니다.") from error
    else:
        tokens = [token for token in re.split(r"[\s,;]+", raw_data.strip()) if token]
    try:
        values = [float(token) for token in tokens]
    except ValueError as error:
        raise ValueError("CSV/텍스트에는 숫자 가중치만 입력하세요.") from error
    if not values or any(not math.isfinite(value) for value in values):
        raise ValueError("전역 모델 가중치는 비어 있지 않은 유한한 숫자여야 합니다.")
    return values


def _reshape_flat_values(values: list[float], reference: dict) -> dict:
    expected_count = sum(torch.as_tensor(item).numel() for item in reference.values())
    if len(values) != expected_count:
        raise ValueError(
            f"가중치 개수가 맞지 않습니다: {len(values)}개 입력, {expected_count}개 필요"
        )
    flat = torch.tensor(values, dtype=torch.float32)
    result = {}
    offset = 0
    for name, reference_values in reference.items():
        reference_tensor = torch.as_tensor(reference_values)
        count = reference_tensor.numel()
        result[name] = flat[offset : offset + count].reshape(reference_tensor.shape).tolist()
        offset += count
    return result
