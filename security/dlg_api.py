"""실제 SmallCNN DLG REST 연결에서 사용하는 순수 연산 함수.

- 클라이언트: 자신의 CIFAR-10 이미지로 1회 그래디언트 생성 후 선택적 가우시안 노이즈 추가
- 서버: 그래디언트와 공유 전역 모델만으로 DLG 이미지 복원
- 평가용 원본은 서버 측 복원 함수에 전달하지 않음
"""
from __future__ import annotations

import math
import torch
from common.model import build_model
from security.dlg import compute_shared_gradients, reconstruct_from_gradients
from security.gradient_noise import add_gaussian_noise

IMAGE_SHAPE = (3, 32, 32)


def make_gradient_payload(global_model: dict, image: torch.Tensor, label: int,
                          *, sigma: float = 0.0, seed: int = 42) -> dict:
    """image는 이미 SmallCNN 훈련과 동일한 -1~1 정규화를 거친 [3,32,32] 텐서."""
    if tuple(image.shape) != IMAGE_SHAPE:
        raise ValueError('단일 CIFAR-10 이미지 텐서 [3,32,32]가 필요합니다.')
    if not 0 <= label <= 9:
        raise ValueError('CIFAR-10 라벨은 0~9여야 합니다.')
    if not math.isfinite(float(sigma)) or float(sigma) < 0:
        raise ValueError('sigma는 0 이상의 유한한 숫자여야 합니다.')
    torch.set_num_threads(1)
    model = build_model(global_model).eval()
    original = compute_shared_gradients(model, image.detach().float().unsqueeze(0), label)
    shared = add_gaussian_noise(original, std=float(sigma), seed=seed)
    # 각 파라미터 텐서 형태를 유지하여 서버가 검증할 수 있게 한다.
    return {
        'format': 'small_cnn_gradient_v1',
        'values': [g.detach().cpu().tolist() for g in shared],
        'noise_sigma': float(sigma),
        'image_shape': list(IMAGE_SHAPE),
    }


def reconstruct_gradient_payload(global_model: dict, gradient: dict,
                                 *, steps: int = 2, seed: int = 42) -> dict:
    """서버 측: 원본 이미지, 정답 레이블 없이 모델과 그래디언트만 사용한다."""
    if gradient.get('format') != 'small_cnn_gradient_v1':
        raise ValueError('실제 SmallCNN 그래디언트 형식이 아닙니다.')
    if gradient.get('image_shape') != list(IMAGE_SHAPE):
        raise ValueError('이미지 크기는 [3,32,32]이어야 합니다.')
    if not 1 <= int(steps) <= 1000:
        raise ValueError('DLG 반복 횟수는 1~1000이어야 합니다.')
    torch.set_num_threads(1)
    model = build_model(global_model).eval()
    params = tuple(model.parameters())
    values = gradient.get('values')
    if not isinstance(values, list) or len(values) != len(params):
        raise ValueError('그래디언트 파라미터 개수가 맞지 않습니다.')
    gradients = []
    for param, values_for_param in zip(params, values):
        g = torch.as_tensor(values_for_param, dtype=param.dtype)
        if g.shape != param.shape or not torch.isfinite(g).all():
            raise ValueError('그래디언트 shape 또는 숫자가 올바르지 않습니다.')
        gradients.append(g)
    found = reconstruct_from_gradients(
        model, tuple(gradients), IMAGE_SHAPE,
        steps=int(steps), seed=seed, image_min=-1.0, image_max=1.0,
    )
    # 원본은 0~1 픽셀 범위. 서버 반환값도 0~1 픽셀 범위로 일치시킨다.
    pixels = ((found.image.detach().cpu() + 1.0) * 0.5).clamp(0, 1)
    return {
        'reconstructed_image': pixels.reshape(-1).tolist(),
        'image_shape': list(IMAGE_SHAPE),
        'predicted_label': found.predicted_label,
        'gradient_loss_before': found.match_loss_before,
        'gradient_loss': found.match_loss_after,
        'iterations': int(steps),
    }


def original_pixels(image: torch.Tensor) -> list[float]:
    """평가용 관리자에게만 전달할 0~1 범위 CHW 순서 원본 픽셀."""
    if tuple(image.shape) != IMAGE_SHAPE:
        raise ValueError('단일 CIFAR-10 이미지 텐서 [3,32,32]가 필요합니다.')
    return ((image.detach().cpu().float() + 1.0) * 0.5).clamp(0, 1).reshape(-1).tolist()
