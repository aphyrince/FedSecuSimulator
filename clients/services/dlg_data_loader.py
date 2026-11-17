"""DLG 이미지 및 레이블 파일 로딩 인터페이스."""

import json
from pathlib import Path
from typing import Any


def resolve_shared_path(path: str, project_root: Path) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = project_root / candidate
    return candidate.resolve()


def load_image(path: str, project_root: Path) -> list[float]:
    image_path = resolve_shared_path(path, project_root)
    if not image_path.exists():
        raise FileNotFoundError(f"DLG image file not found: {image_path}")

    suffix = image_path.suffix.lower()
    if suffix == ".json":
        image = json.loads(image_path.read_text(encoding="utf-8"))
        if isinstance(image, dict):
            image = image.get("image", image.get("values"))
        return flatten_image(image)
    if suffix == ".txt":
        return [float(value.strip()) for value in image_path.read_text().split(",") if value.strip()]
    if suffix == ".npy":
        try:
            import numpy as np
        except ImportError as exc:
            raise RuntimeError("NumPy is required for .npy DLG images") from exc
        return flatten_image(np.load(image_path, allow_pickle=False))

    try:
        from PIL import Image
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("Pillow and NumPy are required for image file DLG inputs") from exc
    return flatten_image(np.asarray(Image.open(image_path), dtype=np.float32))


def load_label(path: str, fallback: Any, project_root: Path) -> int:
    if not path:
        return int(fallback)
    label_path = resolve_shared_path(path, project_root)
    if not label_path.exists():
        raise FileNotFoundError(f"DLG label file not found: {label_path}")

    suffix = label_path.suffix.lower()
    if suffix == ".json":
        label_data = json.loads(label_path.read_text(encoding="utf-8"))
        if isinstance(label_data, dict):
            label_data = label_data.get("label", label_data.get("value"))
        return int(label_data)
    if suffix == ".npy":
        try:
            import numpy as np
        except ImportError as exc:
            raise RuntimeError("NumPy is required for .npy DLG labels") from exc
        return int(np.asarray(np.load(label_path, allow_pickle=False)).reshape(-1)[0])
    return int(label_path.read_text(encoding="utf-8").strip().split()[0].rstrip(","))


def flatten_image(image: Any) -> list[float]:
    if hasattr(image, "reshape"):
        image = image.reshape(-1).tolist()
    if isinstance(image, list):
        result = []
        for value in image:
            if isinstance(value, list):
                result.extend(flatten_image(value))
            else:
                result.append(float(value))
        return result
    raise ValueError("Unsupported DLG image payload")
