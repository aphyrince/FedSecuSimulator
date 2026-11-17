"""교육용 단일 이미지 DLG: 관측 그래디언트로 입력과 레이블 분포를 동시에 최적화.

원본 이미지/정답 레이블은 재구성 함수의 입력이 아님.
참고: Zhu et al., NeurIPS 2019 (https://github.com/mit-han-lab/dlg)
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class TinyDLGNet(nn.Module):
    """작은 교육용 분류기. 실제 팀 연합학습 모델과는 별개."""

    def __init__(self, image_size: int = 8, num_classes: int = 10):
        super().__init__()
        if image_size < 4:
            raise ValueError("image_size는 4 이상이어야 합니다.")
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3 * image_size * image_size, num_classes),
        )

    def forward(self, images: Tensor) -> Tensor:
        return self.net(images)


def compute_shared_gradients(
    model: nn.Module, image: Tensor, label: int
) -> tuple[Tensor, ...]:
    """실험용 '피해 클라이언트'에서 한 이미지의 1회 그래디언트를 생성."""
    if image.ndim != 4 or image.size(0) != 1:
        raise ValueError("image는 [1, 3, H, W] 형태의 단일 이미지여야 합니다.")
    params = tuple(model.parameters())
    logits = model(image)
    target = torch.tensor([label], dtype=torch.long, device=logits.device)
    loss = F.cross_entropy(logits, target)
    grads = torch.autograd.grad(loss, params)
    return tuple(g.detach().clone() for g in grads)


@dataclass
class DLGResult:
    image: Tensor
    predicted_label: int
    match_loss_before: float
    match_loss_after: float
    history: list[dict[str, float | int]]


def reconstruct_from_gradients(
    model: nn.Module,
    observed_gradients: tuple[Tensor, ...],
    image_shape: tuple[int, int, int],
    num_classes: int = 10,
    steps: int = 100,
    lr: float = 0.1,
    seed: int = 42,
    image_min: float = 0.0,
    image_max: float = 1.0,
) -> DLGResult:
    """공격자 측: 모델과 공유 그래디언트만으로 이미지/레이블 추정.

    원본 이미지나 정답 레이블을 전달하지 않으며, MSE도 공격의 손실함수로 사용하지 않는다.
    image_min/max 기본값은 기존 0~1 실험과 호환된다.
    실제 CIFAR-10 SmallCNN의 정규화 입력은 -1~1이므로 호출 시 bounds를 지정한다.
    """
    if steps < 1 or lr <= 0:
        raise ValueError("steps >= 1, lr > 0 이어야 합니다.")
    if not (float('-inf') < image_min < image_max < float('inf')):
        raise ValueError("image_min < image_max 이고 두 값은 유한해야 합니다.")
    if len(image_shape) != 3 or image_shape[0] != 3:
        raise ValueError("image_shape은 (3, 높이, 너비)여야 합니다.")
    if num_classes < 2:
        raise ValueError("num_classes는 2 이상이어야 합니다.")
    params = tuple(model.parameters())
    if not params or len(observed_gradients) != len(params):
        raise ValueError("관측 그래디언트 수가 모델 파라미터와 다릅니다.")
    for p, g in zip(params, observed_gradients):
        if p.shape != g.shape or p.device != g.device:
            raise ValueError("그래디언트의 모양 또는 장치가 모델과 다릅니다.")

    model.eval()
    device, dtype = params[0].device, params[0].dtype
    generator = torch.Generator(device=device).manual_seed(seed)
    dummy_image = nn.Parameter(
        torch.rand((1, *image_shape), device=device, dtype=dtype, generator=generator)
        * (image_max - image_min) + image_min
    )
    dummy_label_logits = nn.Parameter(
        torch.zeros((1, num_classes), device=device, dtype=dtype)
    )
    optimizer = torch.optim.Adam([dummy_image, dummy_label_logits], lr=lr)
    history: list[dict[str, float | int]] = []

    # 모델 파라미터는 고정된 상태로 사용: 최적화 대상은 임의 이미지와 라벨 로짓뿐.
    # 단, create_graph=True로 2차 미분 그래프를 생성하기 위해 파라미터의 requires_grad=True가 필요.
    best_loss = float('inf')
    best_image = None
    best_label = None

    def evaluate_loss() -> Tensor:
        predicted = model(dummy_image)
        soft_label = F.softmax(dummy_label_logits, dim=1)
        soft_ce = -(soft_label * F.log_softmax(predicted, dim=1)).sum()
        dummy_grads = torch.autograd.grad(soft_ce, params, create_graph=True)
        return sum((g - original_g).square().sum()
                   for g, original_g in zip(dummy_grads, observed_gradients))

    for step in range(steps + 1):
        optimizer.zero_grad(set_to_none=True)
        match_loss = evaluate_loss()
        loss_value = float(match_loss.detach())
        history.append({'step': step, 'gradient_match_loss': loss_value})
        if loss_value < best_loss:
            best_loss = loss_value
            best_image = dummy_image.detach().clone()
            best_label = int(dummy_label_logits.detach().argmax(dim=1).item())
        if step < steps:
            match_loss.backward()
            optimizer.step()
            with torch.no_grad():
                dummy_image.clamp_(image_min, image_max)

    assert best_image is not None and best_label is not None
    return DLGResult(
        image=best_image,
        predicted_label=best_label,
        match_loss_before=history[0]['gradient_match_loss'],
        match_loss_after=best_loss,
        history=history,
    )
