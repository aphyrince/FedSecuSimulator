"""DLG 공개 가상 구현과 팀 교체용 인터페이스."""

from copy import deepcopy
import random
from typing import Any


def create_dummy_image(
    global_model: dict[str, Any],
    image_shape: list[int],
    seed: int,
) -> list[float]:
    """DLG 최적화의 시작점(dummy image) 생성 교체 지점."""
    del global_model
    size = max(1, int(__import__("math").prod(image_shape)))
    rng = random.Random(seed)
    return [rng.random() for _ in range(size)]


def reconstruct_image(
    global_model: dict[str, Any],
    target_gradient: dict[str, Any],
    image_shape: list[int],
    label: int,
    seed: int,
    iterations: int = 200,
    dummy_image: list[float] | None = None,
) -> dict[str, Any]:
    """경사 차이를 최소화해 단일 이미지를 복원하는 소규모 DLG 구현."""
    target = target_gradient.get("values", [])
    size = max(1, int(__import__("math").prod(image_shape)))
    if not target:
        target = [0.0] * size

    weights = global_model.get("weights", [1.0] * size)
    flat_weights = weights if isinstance(weights, list) else list(weights.values())
    if len(flat_weights) < size:
        flat_weights = (flat_weights * ((size // len(flat_weights)) + 1))[:size]

    dummy = list(dummy_image) if dummy_image is not None else create_dummy_image(
        global_model, image_shape, seed
    )
    final_loss = 0.0
    learning_rate = 0.05
    for _ in range(max(1, iterations)):
        gradient = [
            dummy[index] * flat_weights[index]
            + label / 10.0
            for index in range(size)
        ]
        difference = [
            gradient[index] - target[index % len(target)]
            for index in range(size)
        ]
        final_loss = sum(value * value for value in difference) / size
        for index in range(size):
            dummy[index] -= learning_rate * difference[index] * flat_weights[index]
            dummy[index] = min(1.0, max(0.0, dummy[index]))
    return {
        "reconstructed_image": dummy,
        "image_shape": image_shape,
        "gradient_loss": round(final_loss, 8),
        "iterations": iterations,
    }
