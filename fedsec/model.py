"""Shared CIFAR-10 CNN, also reusable by the DLG implementation."""

import torch
from torch import nn
from .config import ModelConfig


class SmallCNN(nn.Module):
    """[N, 3, 32, 32] -> [N, 10] logits; no BatchNorm integer buffers."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        first, second = config.channels
        self.features = nn.Sequential(
            nn.Conv2d(3, first, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(first, second, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(second * 8 * 8, config.hidden_size),
            nn.ReLU(), nn.Linear(config.hidden_size, 10),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(images))


def build_model(config: ModelConfig, state=None) -> SmallCNN:
    model = SmallCNN(config)
    if state is not None:
        from .tensors import validate_state
        validate_state(state, model.state_dict())
        model.load_state_dict(state, strict=True)
    return model
