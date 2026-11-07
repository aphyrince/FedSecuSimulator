
import torch

from security.defenses import clip_update, coordinate_median


def test_clip_update():
    update = [torch.tensor([3.0, 4.0])]

    result = clip_update(update, max_norm=2.0)

    expected = torch.tensor([1.2, 1.6])

    assert torch.allclose(result[0], expected)
    assert torch.allclose(
        update[0], torch.tensor([3.0, 4.0])
    )


def test_clip_without_change():
    update = [torch.tensor([3.0, 4.0])]

    result = clip_update(update, max_norm=10.0)

    assert torch.allclose(result[0], update[0])


def test_median_odd():
    updates = [
        [torch.tensor([1.0, 8.0])],
        [torch.tensor([3.0, 4.0])],
        [torch.tensor([9.0, 2.0])],
    ]

    result = coordinate_median(updates)

    assert torch.allclose(
        result[0], torch.tensor([3.0, 4.0])
    )


def test_median_even():
    updates = [
        [torch.tensor([1.0, 8.0])],
        [torch.tensor([3.0, 4.0])],
        [torch.tensor([9.0, 2.0])],
        [torch.tensor([11.0, 12.0])],
    ]

    result = coordinate_median(updates)

    assert torch.allclose(
        result[0], torch.tensor([6.0, 6.0])
    )
