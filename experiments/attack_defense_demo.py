"""10개 가상 클라이언트의 공격/방어 계산 검증용 작은 실험.

주의: CIFAR-10을 학습하지 않으며, 출력값은 정확도가 아닌 가중치 업데이트 예제다.
실행: 프로젝트 최상위 폴더에서 python -m experiments.attack_defense_demo
"""

import csv
from pathlib import Path

import torch

from security.defenses import clip_update, coordinate_median

CLIP_MAX_NORM = 0.7          # 장난감 예제용. 최종 실험 값으로 확정한 것은 아님
ATTACK_MULTIPLIER = -5.0
ATTACKER_IDS = (8, 9)       # 클라이언트 인덱스, 0~9
OUTPUT_DIR = Path('results') / 'attack_defense_demo'


# 이 파일에서만 사용하는 임시 FedAvg. 추후 1번 담당자의 FedAvg로 교체한다.
def local_fedavg(client_updates, sample_counts):
    if len(client_updates) != len(sample_counts) or not client_updates:
        raise ValueError('업데이트와 표본 수를 확인하세요.')
    total_samples = sum(sample_counts)
    if total_samples <= 0 or any(c <= 0 for c in sample_counts):
        raise ValueError('표본 수는 양수여야 합니다.')

    result = []
    for layer in range(len(client_updates[0])):
        avg = torch.zeros_like(client_updates[0][layer])
        for update, count in zip(client_updates, sample_counts):
            avg += update[layer] * (count / total_samples)
        result.append(avg)
    return result


def update_l2_norm(update):
    return sum(layer.detach().double().square().sum().item()
               for layer in update) ** 0.5


def distance_between(a, b):
    return sum((x - y).detach().double().square().sum().item()
               for x, y in zip(a, b)) ** 0.5


def make_updates():
    """10개의 학습 결과가 이미 있다고 가정한 작은 수치 예제."""
    values = [
        [0.40, 0.45], [0.43, 0.48], [0.42, 0.47], [0.45, 0.50],
        [0.46, 0.49], [0.40, 0.46], [0.44, 0.47], [0.41, 0.48],
        [0.42, 0.47], [0.45, 0.49],
    ]
    clean = [[torch.tensor(v, dtype=torch.float64)] for v in values]
    sample_counts = [100] * 10

    # 명세: 참여 악성 클라이언트의 *정상* 업데이트 평균에 -5를 곱해,
    # 해당 악성 클라이언트들이 동일한 업데이트를 제출한다.
    malicious_mean = torch.stack(
        [clean[i][0] for i in ATTACKER_IDS]
    ).mean(dim=0)
    malicious_update = ATTACK_MULTIPLIER * malicious_mean

    attacked = [[t.clone() for t in update] for update in clean]
    for i in ATTACKER_IDS:
        attacked[i] = [malicious_update.clone()]

    return clean, attacked, sample_counts


def aggregate(updates, sample_counts, defense):
    if defense == 'fedavg':
        return local_fedavg(updates, sample_counts)

    clipped = [clip_update(update, CLIP_MAX_NORM) for update in updates]
    if defense == 'clipping_fedavg':
        return local_fedavg(clipped, sample_counts)
    if defense == 'clipping_median':
        return coordinate_median(clipped)
    raise ValueError(f'알 수 없는 방어 방식: {defense}')


def run_experiment():
    clean, attacked, counts = make_updates()
    baseline = aggregate(clean, counts, 'fedavg')

    summary = []
    clients = []
    for mode, updates in [('normal', clean), ('attack', attacked)]:
        for client_id, (update, count) in enumerate(zip(updates, counts)):
            clipped = clip_update(update, CLIP_MAX_NORM)
            clients.append({
                'mode': mode,
                'client_id': client_id,
                'designated_attacker': int(client_id in ATTACKER_IDS),
                'num_samples': count,
                'update_0': f'{update[0][0].item():.6f}',
                'update_1': f'{update[0][1].item():.6f}',
                'norm_before_clipping': f'{update_l2_norm(update):.6f}',
                'norm_after_clipping': f'{update_l2_norm(clipped):.6f}',
            })

        for defense in ('fedavg', 'clipping_fedavg', 'clipping_median'):
            result = aggregate(updates, counts, defense)
            summary.append({
                'mode': mode,
                'defense': defense,
                'result_0': f'{result[0][0].item():.6f}',
                'result_1': f'{result[0][1].item():.6f}',
                'distance_to_normal_fedavg': f'{distance_between(result, baseline):.6f}',
                'clip_max_norm': CLIP_MAX_NORM,
                'attack_multiplier': ATTACK_MULTIPLIER,
                'attacker_ids': '8,9',
            })
    return summary, clients


def save_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == '__main__':
    summary_rows, client_rows = run_experiment()
    summary_path = OUTPUT_DIR / 'summary.csv'
    client_path = OUTPUT_DIR / 'client_update_log.csv'
    save_csv(summary_path, summary_rows)
    save_csv(client_path, client_rows)

    print('mode    defense          result_0   result_1   baseline_distance')
    for row in summary_rows:
        print(f"{row['mode']:<7} {row['defense']:<17} "
              f"{row['result_0']:>8}   {row['result_1']:>8}   "
              f"{row['distance_to_normal_fedavg']:>8}")
    print(f'\n결과 저장: {summary_path}, {client_path}')
    print('주의: 이 값은 모델 정확도가 아닌 가상 업데이트 집계 결과입니다.')
