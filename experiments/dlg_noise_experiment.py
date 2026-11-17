"""기존 DLG와 분리된 그래디언트 가우시안 노이즈 방어 비교 실험.

실행 예:
python -m experiments.dlg_noise_experiment --source synthetic --steps 60
python -m experiments.dlg_noise_experiment --source cifar10 --index 25 --steps 200
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import torch
from PIL import Image, ImageDraw

from experiments.dlg_experiment import load_image, save_png
from security.dlg import TinyDLGNet, compute_shared_gradients, reconstruct_from_gradients
from security.gradient_noise import add_gaussian_noise


def _l2_norm(gradients):
    return math.sqrt(sum(float(g.detach().double().square().sum()) for g in gradients))


def _noise_std_tag(std: float) -> str:
    # ex) 0 -> 0, 0.01 -> 0p01, 0.1 -> 0p1
    return f"{std:g}".replace('.', 'p')


def run(
    source: str = 'synthetic',
    steps: int = 60,
    seed: int = 42,
    index: int = 25,
    data_dir: Path = Path('data'),
    out_dir: Path = Path('results/dlg_noise/synthetic'),
    noise_stds: tuple[float, ...] = (0.0, 0.01, 0.1),
):
    if not noise_stds or any(not math.isfinite(x) or x < 0 for x in noise_stds):
        raise ValueError("noise_stds에는 0 이상의 유한한 값만 사용해야 합니다.")
    if len(set(noise_stds)) != len(noise_stds):
        raise ValueError("noise_stds에는 중복 값이 없어야 합니다.")
    torch.manual_seed(seed)
    image_size = 8 if source == 'synthetic' else 32
    original, true_label, image_id = load_image(source, image_size, index, data_dir)
    model = TinyDLGNet(image_size=image_size)
    model.eval()
    clean_gradients = compute_shared_gradients(model, original, true_label)
    clean_norm = _l2_norm(clean_gradients)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_png(original, out_dir / 'original.png')

    summaries = []
    comparison_paths = [('Original', out_dir / 'original.png')]
    for std in noise_stds:
        # 동일 모델/이미지/복원 초기값을 유지하고 노이즈 수준만 변경.
        shared_gradients = add_gaussian_noise(clean_gradients, std=std, seed=seed + 2)
        noise_norm = _l2_norm(tuple(a - b for a, b in zip(shared_gradients, clean_gradients)))
        result = reconstruct_from_gradients(
            model=model,
            observed_gradients=shared_gradients,
            image_shape=(3, image_size, image_size),
            num_classes=10,
            steps=steps,
            seed=seed + 1,
        )
        # 원본은 오직 복원 뒤 평가에만 사용.
        mse = float((original - result.image).square().mean().item())
        tag = _noise_std_tag(std)
        reconstructed_path = out_dir / f'reconstructed_std_{tag}.png'
        save_png(result.image, reconstructed_path)
        comparison_paths.append((f'std={std:g}', reconstructed_path))
        with (out_dir / f'loss_history_std_{tag}.csv').open('w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=['step', 'gradient_match_loss'])
            writer.writeheader()
            writer.writerows(result.history)
        summaries.append({
            'image_id': image_id,
            'model': 'TinyDLGNet-linear',
            'seed': seed,
            'steps': steps,
            'noise_std': std,
            'clean_gradient_l2': round(clean_norm, 8),
            'added_noise_l2': round(noise_norm, 8),
            'noise_to_gradient_ratio': round(noise_norm / clean_norm, 8) if clean_norm > 0 else None,
            'gradient_match_loss_before': round(result.match_loss_before, 8),
            'gradient_match_loss_after': round(result.match_loss_after, 8),
            'mse_for_evaluation_only': round(mse, 8),
            'true_label_for_evaluation_only': true_label,
            'predicted_label': result.predicted_label,
        })
        print(f'noise_std={std:g} | mse={mse:.6f} | predicted_label={result.predicted_label} | gradient_match_loss={result.match_loss_after:.6f}')

    with (out_dir / 'summary.csv').open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    (out_dir / 'summary.json').write_text(
        json.dumps(summaries, indent=2, ensure_ascii=False), encoding='utf-8'
    )
    # 보고서/발표용 비교 이미지: 원본+노이즈 3단계 나란히
    scale = 12 if image_size == 8 else 6
    margin = 8
    label_height = 20
    cell_size = image_size * scale
    collage = Image.new('RGB', ((cell_size + margin) * len(comparison_paths), cell_size + label_height), 'white')
    draw = ImageDraw.Draw(collage)
    for i, (title, path) in enumerate(comparison_paths):
        with Image.open(path) as im:
            thumbnail = im.convert('RGB').resize((cell_size, cell_size), Image.Resampling.NEAREST)
        x = i * (cell_size + margin)
        collage.paste(thumbnail, (x, label_height))
        draw.text((x + 2, 3), title, fill='black')
    collage.save(out_dir / 'comparison.png')

    print('결과 파일:', out_dir)
    return summaries


def main():
    parser = argparse.ArgumentParser(description='DLG 가우시안 노이즈 방어 비교 실험')
    parser.add_argument('--source', choices=['synthetic', 'cifar10'], default='synthetic')
    parser.add_argument('--steps', type=int, default=60)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--index', type=int, default=25)
    parser.add_argument('--data-dir', type=Path, default=Path('data'))
    parser.add_argument('--out-dir', type=Path, default=None)
    parser.add_argument('--noise-stds', nargs='+', type=float, default=[0.0, 0.01, 0.1])
    args = parser.parse_args()
    out_dir = args.out_dir or Path('results/dlg_noise') / args.source
    run(args.source, args.steps, args.seed, args.index, args.data_dir, out_dir, tuple(args.noise_stds))


if __name__ == '__main__':
    main()
