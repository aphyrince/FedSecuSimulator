"""연합학습과 REST에서 기존에 사용하던 CIFAR-10 기본 모델."""

from copy import deepcopy

from torch import nn

from fedsec.validation import dropout_probability, positive_int

MODEL_FORMAT = "small_cnn_v1"
DEFAULT_MODEL_CONFIG = {"channels": [32, 64], "hidden_size": 128}


def small_cnn_config(values=None) -> dict:
    values = {} if values is None else values
    if not isinstance(values, dict):
        raise ValueError("model_config must be an object")
    if values.get("architecture", MODEL_FORMAT) != MODEL_FORMAT:
        raise ValueError("Unsupported or mismatched model architecture")
    options = {key: value for key, value in values.items() if key != "architecture"}
    if options.keys() - {*DEFAULT_MODEL_CONFIG, "dropout"}:
        raise ValueError("SmallCNN model_config supports channels, hidden_size and dropout")
    result = {**deepcopy(DEFAULT_MODEL_CONFIG), **options}
    channels = result["channels"]
    if not isinstance(channels, (list, tuple)) or len(channels) != 2:
        raise ValueError("channels must contain two positive integers")
    result["channels"] = [positive_int(channel, "channels") for channel in channels]
    positive_int(result["hidden_size"], "hidden_size")
    if "dropout" in result:
        result["dropout"] = dropout_probability(result["dropout"])
    return result


class SmallCNN(nn.Module):
    """32×32 RGB 이미지를 받아 10개 클래스 점수를 반환한다."""

    def __init__(self, config=None):
        super().__init__()
        config = small_cnn_config(config)
        first, second = config["channels"]
        self.features = nn.Sequential(
            nn.Conv2d(3, first, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(first, second, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(second * 8 * 8, config["hidden_size"]),
            nn.ReLU(), nn.Linear(config["hidden_size"], 10),
        )
        if config.get("dropout", 0.0):
            self.classifier[2] = nn.Sequential(self.classifier[2], nn.Dropout(config["dropout"]))

    def forward(self, images):
        return self.classifier(self.features(images))
