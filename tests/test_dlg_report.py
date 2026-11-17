from exp_manager.services.dlg_report import save_dlg_images


def test_evaluation_images(tmp_path):
    files = save_dlg_images([0.0] * 3072, [1.0] * 3072, tmp_path)
    assert set(files) == {'original', 'reconstructed', 'comparison'}
    assert all((tmp_path / name).is_file() for name in files.values())
