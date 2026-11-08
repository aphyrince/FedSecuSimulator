"""REST 모델 직렬화와 체크포인트 처리. 모델 구조는 fedsec.model에서 정의한다."""

from copy import deepcopy
import hashlib
from pathlib import Path
import torch
from fedsec.model import MODEL_FORMAT, DEFAULT_MODEL_CONFIG, PREPROCESSING, SmallCNN, model_config
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


def build_model(payload: dict) -> SmallCNN:
    if payload.get("architecture") != MODEL_FORMAT:
        raise ValueError("This operation requires a small_cnn_v1 model")
    if payload.get("preprocessing") != PREPROCESSING:
        raise ValueError("Model preprocessing is incompatible with CIFAR-10 loading")
    model = SmallCNN(payload.get("model_config"))
    model.load_state_dict(decode_weights(payload.get("weights"), model.state_dict()), strict=True)
    return model


def create_model(seed: int = 42, config=None) -> dict:
    positive_int(seed, "seed", 0)
    config = model_config(config)
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        model = SmallCNN(config)
    weights = encode_weights(model.state_dict())
    return {"model_version": 0, "architecture": MODEL_FORMAT, "model_config": config,
            "preprocessing": deepcopy(PREPROCESSING), "seed": seed,
            "num_parameters": sum(p.numel() for p in model.parameters()), "weights": weights,
            "initial_model_hash": weights_hash(weights)}


def save_model_checkpoint(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    state = decode_weights(payload["weights"])
    torch.save({"format_version": 1, **{k: v for k, v in payload.items() if k not in ("weights", "training_config", "update_deadline")},
                "weights": state, "model_hash": weights_hash(state)}, path)


def load_model_checkpoint(path: str | Path) -> dict:
    payload = torch.load(path, weights_only=True, map_location="cpu")
    if payload.get("format_version") != 1 or weights_hash(payload["weights"]) != payload.get("model_hash"):
        raise ValueError("Invalid model checkpoint")
    payload["weights"] = encode_weights(payload["weights"])
    build_model(payload)
    return payload
