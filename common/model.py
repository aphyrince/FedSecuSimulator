"""REST 모델 직렬화와 체크포인트 처리. 모델 구조는 fedsec.model에서 정의한다."""

from copy import deepcopy
import hashlib
from pathlib import Path
import torch
from fedsec.checkpoint import load_checkpoint
from fedsec.model import (
    DEFAULT_MODEL_CONFIG, MODEL_FORMAT, PREPROCESSING, SUPPORTED_ARCHITECTURES,
    SmallCNN, create_model as create_torch_model, model_config,
)
from fedsec.validation import positive_int, finite_number


def decode_weights(weights: dict, reference: dict | None = None) -> dict:
    if not isinstance(weights, dict) or not weights:
        raise ValueError("weights must be a nonempty state dictionary")
    if reference is not None and set(weights) != set(reference):
        raise ValueError("Model parameter names do not match")
    result = {}
    for name, values in weights.items():
        if not isinstance(name, str):
            raise ValueError("Parameter names must be strings")
        try:
            tensor = torch.as_tensor(values)
            if not tensor.is_floating_point():
                raise ValueError("Model values must be floating point")
            tensor = tensor.detach().to(device="cpu", dtype=torch.float32).clone()
        except (TypeError, RuntimeError) as error:
            raise ValueError(f"Invalid parameter: {name}") from error
        if not tensor.numel() or not torch.isfinite(tensor).all().item():
            raise ValueError(f"Empty or nonfinite parameter: {name}")
        if reference is not None and tensor.shape != reference[name].shape:
            raise ValueError(f"Parameter shape mismatch: {name}")
        result[name] = tensor
    return result


def encode_weights(state: dict) -> dict:
    return {name: tensor.tolist() for name, tensor in decode_weights(state).items()}


def weights_hash(weights: dict) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(decode_weights(weights).items()):
        digest.update(f"{name}:{tuple(tensor.shape)}:float32:".encode())
        digest.update(tensor.contiguous().numpy().tobytes())
    return digest.hexdigest()


def weights_norm(weights: dict) -> float:
    return sum(t.double().square().sum().item() for t in decode_weights(weights).values()) ** 0.5


def build_model(payload: dict) -> torch.nn.Module:
    architecture = payload.get("architecture")
    if architecture not in SUPPORTED_ARCHITECTURES:
        raise ValueError("Unsupported model architecture")
    if payload.get("preprocessing") != PREPROCESSING:
        raise ValueError("Model preprocessing is incompatible with CIFAR-10 loading")
    config = model_config(payload.get("model_config"), architecture=architecture)
    model = create_torch_model(config=config)
    model.load_state_dict(decode_weights(payload.get("weights"), model.state_dict()), strict=True)
    return model


def create_model(seed: int = 42, config=None) -> dict:
    config = model_config(config)
    model = create_torch_model(seed, config)
    weights = encode_weights(model.state_dict())
    return {"model_version": 0, "architecture": config.get("architecture", MODEL_FORMAT), "model_config": config,
            "preprocessing": deepcopy(PREPROCESSING), "seed": seed,
            "num_parameters": sum(p.numel() for p in model.parameters()), "weights": weights,
            "initial_model_hash": weights_hash(weights)}


def save_model_checkpoint(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    state = decode_weights(payload["weights"])
    torch.save({"format_version": 1, **{k: v for k, v in payload.items() if k not in ("weights", "training_config", "update_deadline")},
                "weights": state, "model_hash": weights_hash(state)}, path)


def load_model_checkpoint(path: str | Path) -> dict:
    payload = load_checkpoint(path)
    payload["weights"] = encode_weights(payload["weights"])
    return payload
