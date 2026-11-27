import torch

from fedsec.contracts import ClientUpdate
from fedsec.aggregation import (
    aggregate_fedavg,
    clip_client_updates,
    aggregate_coordinate_median,
)


def test_standalone_clipping():
    base = {"w": torch.tensor([0.0, 0.0])}

    updates = [
        ClientUpdate(0, 1, {
            "w": torch.tensor([3.0, 4.0])
        }),
        ClientUpdate(1, 3, {
            "w": torch.tensor([0.0, 1.0])
        }),
    ]

    clipped = clip_client_updates(base, updates, 2.5)

    assert torch.allclose(
        clipped[0].delta["w"],
        torch.tensor([1.5, 2.0])
    )

    assert torch.allclose(
        clipped[1].delta["w"],
        torch.tensor([0.0, 1.0])
    )

    assert torch.allclose(
        updates[0].delta["w"],
        torch.tensor([3.0, 4.0])
    )

    result = aggregate_fedavg(base, clipped)

    assert torch.allclose(
        result["w"],
        torch.tensor([0.375, 1.25])
    )


def test_standalone_coordinate_median():
    base = {"w": torch.tensor([10.0, 10.0])}

    updates = [
        ClientUpdate(0, 1, {
            "w": torch.tensor([1.0, 8.0])
        }),
        ClientUpdate(1, 1, {
            "w": torch.tensor([2.0, 4.0])
        }),
        ClientUpdate(2, 1, {
            "w": torch.tensor([3.0, 0.0])
        }),
        ClientUpdate(3, 100, {
            "w": torch.tensor([100.0, -100.0])
        }),
    ]

    result = aggregate_coordinate_median(base, updates)

    assert torch.allclose(
        result["w"],
        torch.tensor([12.5, 12.0])
    )
