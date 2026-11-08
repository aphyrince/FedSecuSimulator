"""1번 담당: 독립 실행과 REST 호출에서 함께 사용하는 CIFAR-10 모델 구조."""

from copy import deepcopy
import torch
from torch import nn
from fedsec.validation import positive_int

MODEL_FORMAT = "small_cnn_v1"
DEFAULT_MODEL_CONFIG = {"channels": [32, 64], "hidden_size": 128}
PREPROCESSING = {"mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5]}


def model_config(values=None) -> dict:
    values = {} if values is None else values
    if not isinstance(values, dict) or set(values) - set(DEFAULT_MODEL_CONFIG):
        raise ValueError("model_config supports channels and hidden_size")
    result = {**deepcopy(DEFAULT_MODEL_CONFIG), **values}
    channels = result["channels"]
    if not isinstance(channels, (list, tuple)) or len(channels) != 2:
        raise ValueError("channels must contain two positive integers")
    result["channels"] = [positive_int(c, "channels") for c in channels]
    positive_int(result["hidden_size"], "hidden_size")
    return result


class SmallCNN(nn.Module):
    """32×32 RGB 이미지를 받아 10개 클래스 점수를 출력한다. 서버와 클라이언트가 같은 구조를 사용한다."""

    def __init__(self, config=None):
        super().__init__()
        config = model_config(config)
        first, second = config["channels"]
        self.features = nn.Sequential(
            nn.Conv2d(3, first, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(first, second, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(second * 8 * 8, config["hidden_size"]),
            nn.ReLU(), nn.Linear(config["hidden_size"], 10),
        )

    def forward(self, images):
        return self.classifier(self.features(images))


def create_model(seed: int = 42, config=None) -> SmallCNN:
    positive_int(seed, "seed", 0)
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        return SmallCNN(config)
