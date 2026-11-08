"""DLG 그래디언트 노이즈 방어 함수 및 결과 파일 검증."""

import torch
import pytest

from security.gradient_noise import add_gaussian_noise


def test_zero_noise_returns_same_values_without_sharing_storage():
    original = (torch.tensor([1.0, 2.0]), torch.tensor([[3.0]]))
    result = add_gaussian_noise(original, std=0.0)
    assert all(torch.equal(a, b) for a, b in zip(original, result))
    assert all(a.data_ptr() != b.data_ptr() for a, b in zip(original, result))


def test_seed_reproducibility_and_no_mutation():
    grad = (torch.ones(200), torch.zeros(40))
    initial = tuple(g.clone() for g in grad)
    a = add_gaussian_noise(grad, std=0.1, seed=10)
    b = add_gaussian_noise(grad, std=0.1, seed=10)
    c = add_gaussian_noise(grad, std=0.1, seed=11)
    assert all(torch.equal(x, y) for x, y in zip(a, b))
    assert not torch.equal(a[0], c[0])
    assert all(torch.equal(x, y) for x, y in zip(grad, initial))
    assert a[0].shape == grad[0].shape and a[0].dtype == grad[0].dtype


@pytest.mark.parametrize('std', [-1.0, float('nan'), float('inf')])
def test_rejects_invalid_std(std):
    with pytest.raises(ValueError):
        add_gaussian_noise((torch.ones(3),), std=std)


def test_rejects_empty_gradients():
    with pytest.raises(ValueError):
        add_gaussian_noise((), std=0.1)


def test_comparison_saves_all_files_and_baseline_is_unchanged(tmp_path):
    from experiments.dlg_noise_experiment import run
    from experiments.dlg_experiment import run as old_dlg_run

    summary = run(
        source='synthetic', steps=3, seed=42, index=25,
        data_dir=tmp_path / 'data', out_dir=tmp_path / 'noise',
        noise_stds=(0.0, 0.01, 0.1),
    )
    baseline = old_dlg_run('synthetic', 3, 42, 25, tmp_path / 'data', tmp_path / 'baseline')
    assert len(summary) == 3
    assert [row['noise_std'] for row in summary] == [0.0, 0.01, 0.1]
    assert summary[0]['mse_for_evaluation_only'] == pytest.approx(
        baseline['mse_for_evaluation_only'], abs=1e-7
    )
    assert summary[0]['added_noise_l2'] == 0.0
    assert summary[1]['added_noise_l2'] > 0
    for name in ('original.png', 'comparison.png', 'summary.csv', 'summary.json',
                 'reconstructed_std_0.png', 'reconstructed_std_0p01.png', 'reconstructed_std_0p1.png',
                 'loss_history_std_0.csv', 'loss_history_std_0p01.csv', 'loss_history_std_0p1.csv'):
        assert (tmp_path / 'noise' / name).exists(), name
