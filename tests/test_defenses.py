
import torch

from security.defenses import (
    clip_update,
    coordinate_median,
    add_gaussian_noise,
)


def test_clip_update():
    update = {
        "weight": torch.tensor([3.0, 4.0])
    }

    result = clip_update(update, max_norm=2.0)

    expected = torch.tensor([1.2, 1.6])

    assert torch.allclose(
        result["weight"], expected
    )


def test_coordinate_median():
    updates = [
        {"weight": torch.tensor([1.0])},
        {"weight": torch.tensor([2.0])},
        {"weight": torch.tensor([100.0])},
    ]

    result = coordinate_median(updates)

    assert result["weight"].item() == 2.0


def test_gaussian_noise_zero():
    gradients = {
        "weight": torch.tensor([1.0, 2.0])
    }

    result = add_gaussian_noise(
        gradients, sigma=0.0
    )

    assert torch.equal(
        result["weight"],
        gradients["weight"]
    )
