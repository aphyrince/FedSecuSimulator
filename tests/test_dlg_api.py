import pytest
import torch
from common.model import create_model
from security.dlg_api import make_gradient_payload, reconstruct_gradient_payload, original_pixels


def test_cnn_gradient_shape_and_reconstruction():
    model = create_model(seed=42)
    image = torch.zeros((3, 32, 32))
    gradient = make_gradient_payload(model, image, 3, sigma=0.0, seed=42)
    assert gradient['format'] == 'small_cnn_gradient_v1'
    assert gradient['image_shape'] == [3, 32, 32]
    result = reconstruct_gradient_payload(model, gradient, steps=1, seed=44)
    assert len(result['reconstructed_image']) == 3072
    assert all(0 <= x <= 1 for x in result['reconstructed_image'])
    assert result['gradient_loss_before'] >= 0
    assert result['gradient_loss'] >= 0
    assert original_pixels(image) == [0.5] * 3072


def test_gaussian_noise_and_validation():
    model = create_model(seed=42)
    image = torch.zeros((3, 32, 32))
    clean = make_gradient_payload(model, image, 3, sigma=0.0, seed=4)
    noisy = make_gradient_payload(model, image, 3, sigma=0.001, seed=4)
    assert clean['values'][0] != noisy['values'][0]
    with pytest.raises(ValueError):
        make_gradient_payload(model, image, 3, sigma=-1.0)
    with pytest.raises(ValueError):
        reconstruct_gradient_payload(model, {'format':'x'}, steps=1)
