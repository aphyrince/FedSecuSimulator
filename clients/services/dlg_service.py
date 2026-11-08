"""DLG 단일 이미지 그래디언트 생성 및 가우시안 노이즈 구현."""

from copy import deepcopy
import random
from typing import Any


def compute_single_image_gradient(
    model: dict[str, Any],
    image: list[float],
    label: int,
) -> dict[str, Any]:
    """가상 선형 모델로 단일 이미지에 대한 그래디언트를 계산한다."""
    weights = model.get("weights", [1.0] * len(image))
    flat_weights = weights if isinstance(weights, list) else list(weights.values())
    if not flat_weights:
        flat_weights = [1.0] * len(image)
    gradient = [
        float(pixel) * float(flat_weights[index % len(flat_weights)])
        + label / 10.0
        for index, pixel in enumerate(image)
    ]
    return {"values": gradient}


def add_gaussian_noise(
    gradient: dict[str, Any],
    sigma: float,
    seed: int,
) -> dict[str, Any]:
    """전송 전 그래디언트 값에 N(0, sigma^2) 노이즈를 추가한다."""
    rng = random.Random(seed)
    values = gradient.get("values", [])
    noisy = [float(value) + rng.gauss(0.0, max(0.0, sigma)) for value in values]
    result = deepcopy(gradient)
    result["values"] = noisy
    result["noise_sigma"] = max(0.0, sigma)
    return result
