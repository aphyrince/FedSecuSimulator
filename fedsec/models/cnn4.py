"""CIFAR-10 모델 비교에 사용하는 합성곱 4층 CNN."""

from torch import nn


CNN4_FORMAT = "cnn4_cifar10_v1"


class CNN4(nn.Module):
    """합성곱 두 층마다 특징 맵을 절반으로 줄이고 10개 클래스 점수를 반환한다."""

    def __init__(self, dropout: float = 0.0):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(128 * 8 * 8, 128), nn.ReLU(), nn.Linear(128, 10),
        )
        if dropout:
            self.classifier[2] = nn.Sequential(self.classifier[2], nn.Dropout(dropout))

    def forward(self, images):
        return self.classifier(self.features(images))
