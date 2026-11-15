"""독립 실행과 REST에서 공용으로 사용하는 모델 선택과 생성."""

import torch
from torch import nn

from fedsec.models.cnn4 import CNN4, CNN4_FORMAT
from fedsec.models.resnet18 import RESNET18_FORMAT, create_resnet18
from fedsec.models.small_cnn import DEFAULT_MODEL_CONFIG, MODEL_FORMAT, SmallCNN, small_cnn_config
from fedsec.validation import dropout_probability, positive_int

SUPPORTED_ARCHITECTURES = (MODEL_FORMAT, CNN4_FORMAT, RESNET18_FORMAT)
PREPROCESSING = {"mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5]}


def model_config(values=None, *, architecture: str | None = None) -> dict:
    values = {} if values is None else values
    if not isinstance(values, dict):
        raise ValueError("model_config must be an object")
    selected = values.get("architecture", architecture if architecture is not None else MODEL_FORMAT)
    if selected not in SUPPORTED_ARCHITECTURES or (architecture is not None and selected != architecture):
        raise ValueError("Unsupported or mismatched model architecture")
    if selected != MODEL_FORMAT:
        if values.keys() - {"architecture", "dropout"}:
            raise ValueError(f"{selected} uses a fixed CIFAR-10 architecture")
        normalized = {"architecture": selected}
        if "dropout" in values:
            normalized["dropout"] = dropout_probability(values["dropout"])
        return normalized
    return small_cnn_config(values)


def create_model(seed: int = 42, config=None) -> nn.Module:
    positive_int(seed, "seed", 0)
    config = model_config(config)
    dropout = config.get("dropout", 0.0)
    with torch.random.fork_rng(devices=[]):
        if config.get("architecture") == RESNET18_FORMAT:
            return create_resnet18(seed, dropout)
        torch.random.default_generator.manual_seed(seed)
        if config.get("architecture") == CNN4_FORMAT:
            return CNN4(dropout)
        return SmallCNN(config)
