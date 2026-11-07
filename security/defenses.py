
import torch


def clip_update(update, max_norm):
    """
    클라이언트의 전체 업데이트 L2 노름을 제한한다.

    update: 가중치 변화량을 담은 텐서 리스트
    max_norm: 클리핑 상한 C
    """

    if not update:
        raise ValueError("업데이트가 비어 있습니다.")

    if max_norm <= 0:
        raise ValueError("max_norm은 0보다 커야 합니다.")

    # 모든 파라미터를 합친 전체 노름 계산
    squared_norm = sum(
        torch.sum(t.detach().double() ** 2).item()
        for t in update
    )
    total_norm = squared_norm ** 0.5

    # 노름이 상한을 넘을 때만 줄이기
    scale = (
        min(1.0, max_norm / total_norm)
        if total_norm > 0 else 1.0
    )

    # 원본 업데이트를 변경하지 않고 새로운 값 반환
    return [t.detach().clone() * scale for t in update]


def coordinate_median(client_updates):
    """
    여러 클라이언트 업데이트의 좌표별 중앙값을 계산한다.

    client_updates:
        [
            [client1_layer1, client1_layer2, ...],
            [client2_layer1, client2_layer2, ...],
            ...
        ]
    """

    if not client_updates or not client_updates[0]:
        raise ValueError("클라이언트 업데이트가 비어 있습니다.")

    n = len(client_updates)
    layers = len(client_updates[0])

    if any(len(update) != layers for update in client_updates):
        raise ValueError("클라이언트별 계층 수가 다릅니다.")

    result = []

    for layer in range(layers):
        values = [update[layer] for update in client_updates]

        if any(t.shape != values[0].shape for t in values):
            raise ValueError("업데이트 텐서의 모양이 다릅니다.")

        # 클라이언트 차원으로 쌓고 좌표마다 정렬
        sorted_values = torch.sort(
            torch.stack(values, dim=0), dim=0
        ).values

        if n % 2 == 1:
            # 홀수: 가운데 값
            median = sorted_values[n // 2].clone()
        else:
            # 짝수: 가운데 두 값의 평균
            median = (
                sorted_values[n // 2 - 1]
                + sorted_values[n // 2]
            ) / 2

        result.append(median)

    return result
