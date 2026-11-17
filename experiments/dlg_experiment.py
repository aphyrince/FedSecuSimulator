"""작은 synthetic 이미지 또는 CIFAR-10 1장으로 DLG 교육용 실험.

최상위 폴더에서 실행:
 python -m experiments.dlg_experiment --source synthetic --steps 60
 python -m experiments.dlg_experiment --source cifar10 --index 25 --steps 200
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch
from PIL import Image

from security.dlg import TinyDLGNet, compute_shared_gradients, reconstruct_from_gradients


def load_image(source: str, image_size: int, index: int, data_dir: Path):
    if source == 'synthetic':
        image = torch.zeros(1, 3, image_size, image_size)
        image[:, 0, image_size // 4:3 * image_size // 4,
              image_size // 4:3 * image_size // 4] = 0.9
        image[:, 1, :, :] = 0.2
        return image, 3, 'synthetic-square'
    if source == 'cifar10':
        try:
            from torchvision import datasets, transforms
        except ImportError as exc:
            raise RuntimeError('CIFAR-10 실험에는 torchvision이 필요합니다. python -m pip install torchvision') from exc
        dataset = datasets.CIFAR10(
            root=str(data_dir), train=True, download=True,
            transform=transforms.ToTensor(),
        )
        if not 0 <= index < len(dataset):
            raise ValueError('CIFAR-10 index가 범위를 벗어났습니다.')
        image, label = dataset[index]
        return image.unsqueeze(0), int(label), f'cifar10-train-{index}'
    raise ValueError(f'지원하지 않는 source: {source}')


def save_png(image: torch.Tensor, path: Path):
    """입력 [1,3,H,W] (0~1) 이미지를 PNG로 저장."""
    array = (image.detach().squeeze(0).cpu().clamp(0, 1)
             .permute(1, 2, 0).mul(255).round().to(torch.uint8).numpy())
    Image.fromarray(array).save(path)


def run(source: str, steps: int, seed: int, index: int, data_dir: Path, out_dir: Path):
    torch.manual_seed(seed)
    image_size = 8 if source == 'synthetic' else 32
    original, true_label, image_id = load_image(source, image_size, index, data_dir)
    model = TinyDLGNet(image_size=image_size)
    model.eval()

    # 실제 환경에서는 원본/정답은 관측 그래디언트를 만든 클라이언트 측에서만 사용.
    observed = compute_shared_gradients(model, original, true_label)
    # 공격자는 원본/정답을 받지 않고 모델과 그래디언트만으로 복원.
    result = reconstruct_from_gradients(
        model=model,
        observed_gradients=observed,
        image_shape=(3, image_size, image_size),
        num_classes=10, steps=steps, seed=seed + 1,
    )

    # 원본과 정답은 공격이 끝난 뒤 평가에만 사용.
    mse = float(torch.mean((original - result.image).square()).item())
    out_dir.mkdir(parents=True, exist_ok=True)
    save_png(original, out_dir / 'original.png')
    save_png(result.image, out_dir / 'reconstructed.png')
    # 2개 이미지를 나란히 붙인 미리보기
    with Image.open(out_dir / 'original.png') as left, Image.open(out_dir / 'reconstructed.png') as right:
        width, height = left.size
        comparison = Image.new('RGB', (width * 2, height))
        comparison.paste(left, (0, 0))
        comparison.paste(right, (width, 0))
        comparison.resize((width * 2 * 16, height * 16), resample=Image.Resampling.NEAREST).save(out_dir / 'comparison.png')
    with (out_dir / 'loss_history.csv').open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=['step', 'gradient_match_loss'])
        writer.writeheader()
        writer.writerows(result.history)

    info = {
        'source': source, 'image_id': image_id,
        'model': 'TinyDLGNet-linear', 'image_shape': list(original.shape),
        'seed': seed, 'steps': steps, 'gradient_noise_std': 0.0,
        'true_label_for_evaluation_only': true_label,
        'predicted_label': result.predicted_label,
        'gradient_match_loss_before': result.match_loss_before,
        'gradient_match_loss_after': result.match_loss_after,
        'mse_for_evaluation_only': mse,
        'torch_version': torch.__version__,
    }
    (out_dir / 'summary.json').write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding='utf-8')
    print('=== DLG reconstruction demonstration ===')
    print('source:', image_id)
    print('gradient_match_loss:', f'{result.match_loss_before:.6f}', '->', f'{result.match_loss_after:.6f}')
    print('MSE (evaluation only):', f'{mse:.6f}')
    print('true label (evaluation only):', true_label)
    print('predicted label:', result.predicted_label)
    print('saved to:', out_dir)
    return info


def main():
    parser = argparse.ArgumentParser(description='단일 이미지 DLG 실험')
    parser.add_argument('--source', choices=['synthetic', 'cifar10'], default='synthetic')
    parser.add_argument('--steps', type=int, default=60)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--index', type=int, default=25)
    parser.add_argument('--data-dir', type=Path, default=Path('data'))
    parser.add_argument('--out-dir', type=Path, default=None)
    args = parser.parse_args()
    out_dir = args.out_dir or (Path('results') / 'dlg' / args.source)
    run(args.source, args.steps, args.seed, args.index, args.data_dir, out_dir)


if __name__ == '__main__':
    main()
