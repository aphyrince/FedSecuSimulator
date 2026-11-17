"""1단계: 실제 SmallCNN + CIFAR-10 단일 이미지 DLG / 가우시안 노이즈 독립 실험.

프로젝트 최상위 폴더에서 실행:
 python -m experiments.dlg_cnn_experiment --steps 2 --noise-stds 0
 python -m experiments.dlg_cnn_experiment --steps 5 --noise-stds 0 0.001
 python -m experiments.dlg_cnn_experiment --checkpoint results/security_baseline_01/checkpoints/global_0001.pt --steps 2 --noise-stds 0

참고: 이 파일은 REST/Flask에 연결하지 않음. 실제 SmallCNN 연결을 먼저 검증하는 단계.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import torch
from PIL import Image, ImageDraw

from common.model import build_model, load_model_checkpoint
from fedsec.data import CifarRangeDataset
from fedsec.model import PREPROCESSING, create_model
from security.dlg import compute_shared_gradients, reconstruct_from_gradients
from security.gradient_noise import add_gaussian_noise


def load_cifar_sample(batch_file: Path, index: int) -> tuple[torch.Tensor, int]:
    """팀의 fedsec.data와 동일한 -1~1 정규화로 CIFAR-10 이미지 1장 로드."""
    batch_file = Path(batch_file).resolve()
    if not batch_file.is_file():
        raise FileNotFoundError(f"CIFAR-10 파일이 없습니다: {batch_file}")
    if index < 0:
        raise ValueError("index는 0 이상이어야 합니다.")
    dataset = CifarRangeDataset({
        'source_dir': str(batch_file.parent),
        'ranges': [{'file': batch_file.name, 'start': index, 'end': index + 1}],
    })
    image, label = dataset[0]
    return image.unsqueeze(0), int(label)


def to_pixel_range(image: torch.Tensor) -> torch.Tensor:
    """SmallCNN 입력 정규화(-1~1)를 원본 픽셀 범위(0~1)로 복원."""
    mean = torch.tensor(PREPROCESSING['mean'], dtype=image.dtype, device=image.device).view(1, 3, 1, 1)
    std = torch.tensor(PREPROCESSING['std'], dtype=image.dtype, device=image.device).view(1, 3, 1, 1)
    return (image * std + mean).clamp(0, 1)


def save_png(image: torch.Tensor, path: Path) -> None:
    image_01 = to_pixel_range(image)
    pixels = image_01.detach().squeeze(0).cpu().permute(1, 2, 0).mul(255).round().to(torch.uint8).numpy()
    Image.fromarray(pixels).save(path)


def norm(gradients: tuple[torch.Tensor, ...]) -> float:
    return math.sqrt(sum(float(g.double().square().sum()) for g in gradients))


def std_tag(value: float) -> str:
    return f"{value:g}".replace('.', 'p')


def run(
    batch_file: Path,
    index: int = 25,
    steps: int = 2,
    noise_stds: tuple[float, ...] = (0.0,),
    seed: int = 42,
    out_dir: Path = Path('results/dlg_cnn_stage1'),
    checkpoint: Path | None = None,
    torch_threads: int = 1,
) -> list[dict]:
    if not noise_stds or any(not math.isfinite(s) or s < 0 for s in noise_stds):
        raise ValueError('noise_stds는 0 이상의 유한한 숫자여야 합니다.')
    if len(set(noise_stds)) != len(noise_stds):
        raise ValueError('noise_stds에 중복이 있습니다.')
    if torch_threads < 1:
        raise ValueError('torch_threads는 1 이상이어야 합니다.')
    torch.set_num_threads(torch_threads)

    original, label = load_cifar_sample(batch_file, index)
    if checkpoint is None:
        model = create_model(seed=seed)
        model_source = 'newly_initialized_small_cnn_v1'
    else:
        model = build_model(load_model_checkpoint(checkpoint))
        model_source = str(checkpoint)
    model.eval()

    clean_grads = compute_shared_gradients(model, original, label)
    clean_norm = norm(clean_grads)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_png(original, out_dir / 'original.png')

    summaries: list[dict] = []
    pictures: list[tuple[str, Path]] = [('Original', out_dir / 'original.png')]
    for std in noise_stds:
        shared = add_gaussian_noise(clean_grads, std=std, seed=seed + 2)
        added_noise = tuple(b - a for a, b in zip(clean_grads, shared))
        # 공격자는 원본 이미지/정답을 입력받지 않고 공유된 그래디언트만으로 복원.
        reconstruction = reconstruct_from_gradients(
            model=model,
            observed_gradients=shared,
            image_shape=(3, 32, 32),
            num_classes=10,
            steps=steps,
            seed=seed + 1,
            image_min=-1.0,
            image_max=1.0,
        )
        # MSE 계산에서는 원본/복원 이미지 모두 0~1 픽셀 값으로 환산.
        mse = float((to_pixel_range(original) - to_pixel_range(reconstruction.image)).square().mean())
        tag = std_tag(std)
        image_path = out_dir / f'reconstructed_std_{tag}.png'
        save_png(reconstruction.image, image_path)
        pictures.append((f'std={std:g}', image_path))

        with (out_dir / f'loss_history_std_{tag}.csv').open('w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=['step', 'gradient_match_loss'])
            writer.writeheader()
            writer.writerows(reconstruction.history)

        summary = {
            'model': 'small_cnn_v1',
            'model_source': model_source,
            'image_index': index,
            'seed': seed,
            'steps': steps,
            'noise_std': std,
            'clean_gradient_norm': clean_norm,
            'noise_norm': norm(added_noise),
            'mse_pixel_0_to_1': mse,
            'gradient_match_loss_before': reconstruction.match_loss_before,
            'gradient_match_loss_after': reconstruction.match_loss_after,
            'true_label_for_evaluation_only': label,
            'predicted_label': reconstruction.predicted_label,
        }
        summaries.append(summary)
        print(f"noise_std={std:g} | MSE={mse:.6f} | loss={reconstruction.match_loss_before:.5g} -> {reconstruction.match_loss_after:.5g}", flush=True)

    with (out_dir / 'summary.csv').open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(summaries[0].keys()))
        writer.writeheader()
        writer.writerows(summaries)
    (out_dir / 'summary.json').write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2), encoding='utf-8'
    )

    scale = 6
    label_height = 23
    w = 32 * scale
    collage = Image.new('RGB', (w * len(pictures), w + label_height), 'white')
    draw = ImageDraw.Draw(collage)
    for i, (title, image_path) in enumerate(pictures):
        with Image.open(image_path) as im:
            thumb = im.convert('RGB').resize((w, w), Image.Resampling.NEAREST)
        collage.paste(thumb, (i * w, label_height))
        draw.text((i * w + 4, 5), title, fill='black')
    collage.save(out_dir / 'comparison.png')
    print(f"완료: {out_dir.resolve()}", flush=True)
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser(description='SmallCNN + CIFAR-10 DLG 독립 실험')
    parser.add_argument('--batch-file', type=Path, default=Path('data/cifar-10-batches-py/test_batch'))
    parser.add_argument('--index', type=int, default=25)
    parser.add_argument('--steps', type=int, default=2)
    parser.add_argument('--noise-stds', nargs='+', type=float, default=[0.0])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--out-dir', type=Path, default=Path('results/dlg_cnn_stage1'))
    parser.add_argument('--torch-threads', type=int, default=1)
    args = parser.parse_args()
    run(
        batch_file=args.batch_file,
        index=args.index,
        steps=args.steps,
        noise_stds=tuple(args.noise_stds),
        seed=args.seed,
        out_dir=args.out_dir,
        checkpoint=args.checkpoint,
        torch_threads=args.torch_threads,
    )


if __name__ == '__main__':
    main()
