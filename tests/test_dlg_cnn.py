"""SmallCNN 기반 DLG 스모크 테스트. 가짜 CIFAR-10 배치로 연결 경로를 검증한다."""
import pickle

import numpy as np
import torch

from fedsec.model import SmallCNN
from experiments.dlg_cnn_experiment import run, to_pixel_range
from security.dlg import compute_shared_gradients, reconstruct_from_gradients


def test_smallcnn_dlg_accepts_normalized_inputs():
    torch.set_num_threads(1)
    model = SmallCNN({'channels': [4, 8], 'hidden_size': 16})
    original = torch.full((1, 3, 32, 32), -0.3)
    observed = compute_shared_gradients(model, original, label=3)
    result = reconstruct_from_gradients(
        model, observed, (3, 32, 32), steps=1,
        seed=42, image_min=-1.0, image_max=1.0,
    )
    assert result.image.shape == original.shape
    assert result.image.min() >= -1 and result.image.max() <= 1
    assert result.match_loss_after <= result.match_loss_before
    assert torch.all((0 <= to_pixel_range(result.image)) & (to_pixel_range(result.image) <= 1))


def test_cnn_experiment_writes_metrics(tmp_path):
    torch.set_num_threads(1)
    # 실제 데이터를 내려받지 않고 로더 형식만 확인한다.
    batch_file = tmp_path / 'test_batch'
    pixels = np.zeros((2, 3072), dtype=np.uint8)
    pixels[1, :] = 150
    batch_file.write_bytes(pickle.dumps({b'data': pixels, b'labels': [1, 4]}))
    output_dir = tmp_path / 'out'
    rows = run(batch_file, index=1, steps=1, noise_stds=(0.0,),
               out_dir=output_dir, torch_threads=1)
    assert len(rows) == 1
    assert rows[0]['model'] == 'small_cnn_v1'
    assert rows[0]['noise_std'] == 0
    assert rows[0]['mse_pixel_0_to_1'] >= 0
    for name in ('original.png', 'reconstructed_std_0.png', 'comparison.png',
                 'loss_history_std_0.csv', 'summary.csv', 'summary.json'):
        assert (output_dir / name).is_file(), name
