"""DLG용 공유 그래디언트에 가우시안 노이즈를 추가하는 실험용 방어.

주의: 단순 노이즈 추가만으로 차등 개인정보보호(DP) 보장을 주장할 수 없다.
"""
from __future__ import annotations

import math
import torch
from torch import Tensor


def add_gaussian_noise(
    gradients: tuple[Tensor, ...], std: float, seed: int = 42
) -> tuple[Tensor, ...]:
    """각 그래디언트 원소에 N(0, std**2) 노이즈를 더한다.

    - 원본 텐서는 수정하지 않음
    - std=0일 때는 별도 복사본을 반환
    - 동일한 seed/입력/환경에서는 동일한 노이즈를 사용
    """
    if not gradients:
        raise ValueError("그래디언트가 비어 있습니다.")
    if not math.isfinite(std) or std < 0:
        raise ValueError("std는 0 이상의 유한한 숫자여야 합니다.")
    if any(not g.is_floating_point() for g in gradients):
        raise TypeError("그래디언트는 실수형 텐서여야 합니다.")
    if std == 0:
        return tuple(g.detach().clone() for g in gradients)

    # 서버에 전송되기 전, 클라이언트 측에서 적용하는 것으로 가정.
    # 그라디언트는 보통 같은 장치에 위치하므로 장치별 난수 생성기를 사용한다.
    generators: dict[torch.device, torch.Generator] = {}
    noisy: list[Tensor] = []
    for g in gradients:
        if g.device not in generators:
            generators[g.device] = torch.Generator(device=g.device).manual_seed(seed)
        noise = torch.randn(
            g.shape,
            device=g.device,
            dtype=g.dtype,
            generator=generators[g.device],
        ) * std
        noisy.append(g.detach().clone() + noise)
    return tuple(noisy)
