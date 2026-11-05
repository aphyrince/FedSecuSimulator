
import torch


# 1. 업데이트 노름 클리핑
def clip_update(update, max_norm):
    if not update:
        raise ValueError("Empty update")
    if max_norm <= 0:
        raise ValueError("max_norm must be positive")

    total_norm = torch.sqrt(sum(
        (value ** 2).sum()
        for value in update.values()
    ))

    scale = min(
        1.0,
        max_norm / (total_norm.item() + 1e-12)
    )

    return {
        name: value * scale
        for name, value in update.items()
    }


# 2. 좌표별 중앙값 집계
def coordinate_median(updates):
    if not updates:
        raise ValueError("Empty updates")

    result = {}

    for name in updates[0]:
        values = torch.stack([
            update[name]
            for update in updates
        ])

        sorted_values = torch.sort(
            values, dim=0
        ).values

        n = len(updates)

        if n % 2 == 1:
            result[name] = sorted_values[n // 2]
        else:
            result[name] = (
                sorted_values[n // 2 - 1]
                + sorted_values[n // 2]
            ) / 2

    return result


# 3. 가우시안 노이즈
def add_gaussian_noise(gradients, sigma):
    if sigma < 0:
        raise ValueError("sigma must be nonnegative")

    return {
        name: value + torch.randn_like(value) * sigma
        for name, value in gradients.items()
    }
