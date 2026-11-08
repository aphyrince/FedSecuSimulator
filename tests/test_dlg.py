"""단일 이미지 DLG의 기초 함수 테스트."""
import torch

from security.dlg import TinyDLGNet, compute_shared_gradients, reconstruct_from_gradients


def test_gradient_shape_matches_model():
    model = TinyDLGNet(image_size=8)
    image = torch.rand((1, 3, 8, 8))
    grads = compute_shared_gradients(model, image, label=2)
    assert len(grads) == len(tuple(model.parameters()))
    assert all(g.shape == p.shape for g, p in zip(grads, model.parameters()))


def test_dlg_improves_gradient_matching_and_preserves_weights():
    torch.manual_seed(7)
    model = TinyDLGNet(image_size=8)
    original = torch.rand((1, 3, 8, 8))
    target_grads = compute_shared_gradients(model, original, label=4)
    original_weights = [p.detach().clone() for p in model.parameters()]

    result = reconstruct_from_gradients(
        model, target_grads, (3, 8, 8), steps=50, seed=31, lr=0.1
    )
    assert result.image.shape == original.shape
    assert result.image.min() >= 0.0 and result.image.max() <= 1.0
    assert result.match_loss_after < result.match_loss_before
    assert len(result.history) == 51
    assert all(torch.equal(p, b) for p, b in zip(model.parameters(), original_weights))


def test_dlg_does_not_require_origin_or_label_for_reconstruction():
    torch.manual_seed(5)
    model = TinyDLGNet(image_size=8)
    grad = compute_shared_gradients(model, torch.rand(1, 3, 8, 8), label=1)
    result = reconstruct_from_gradients(model, grad, (3, 8, 8), steps=2)
    assert result.predicted_label in range(10)


def test_dlg_rejects_wrong_gradient_structure():
    model = TinyDLGNet(image_size=8)
    try:
        reconstruct_from_gradients(model, (torch.zeros(3),), (3, 8, 8), steps=1)
    except ValueError:
        pass
    else:
        raise AssertionError('그래디언트 구조 검증에 실패했습니다.')


def test_experiment_saves_images_and_metrics(tmp_path):
    from experiments.dlg_experiment import run
    info = run(source='synthetic', steps=4, seed=42, index=25,
               data_dir=tmp_path / 'dataset', out_dir=tmp_path / 'result')
    assert info['source'] == 'synthetic'
    assert info['mse_for_evaluation_only'] >= 0
    for file_name in ('original.png', 'reconstructed.png', 'comparison.png',
                      'loss_history.csv', 'summary.json'):
        assert (tmp_path / 'result' / file_name).exists()
