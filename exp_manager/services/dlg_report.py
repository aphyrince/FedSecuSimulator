"""DLG 평가 관리자 전용 이미지 내보내기. 재구성 서버에서는 호출하지 않는다."""
from pathlib import Path
import numpy as np
from PIL import Image


def save_dlg_images(original: list[float], reconstructed: list[float],
                    out_dir: Path) -> dict[str, str]:
    if len(original) != 3072 or len(reconstructed) != 3072:
        raise ValueError('DLG image has to contain 3072 pixels')
    out_dir.mkdir(parents=True, exist_ok=True)

    def _image(values: list[float]) -> Image.Image:
        pixels = np.asarray(values, dtype=np.float32)
        if not np.isfinite(pixels).all() or (pixels < 0).any() or (pixels > 1).any():
            raise ValueError('DLG pixels must be finite and within [0,1]')
        pixels = (pixels.reshape(3, 32, 32).transpose(1, 2, 0) * 255).round().astype(np.uint8)
        return Image.fromarray(pixels, 'RGB')

    before = _image(original)
    after = _image(reconstructed)
    comparison = Image.new('RGB', (64, 32))
    comparison.paste(before, (0, 0))
    comparison.paste(after, (32, 0))
    files = {'original': 'dlg_original.png',
             'reconstructed': 'dlg_reconstructed.png',
             'comparison': 'dlg_comparison.png'}
    before.save(out_dir / files['original'])
    after.save(out_dir / files['reconstructed'])
    comparison.resize((768, 384), resample=Image.Resampling.NEAREST).save(out_dir / files['comparison'])
    return files
