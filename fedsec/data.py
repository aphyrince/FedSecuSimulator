"""매니저가 배정한 CIFAR-10 피클 파일의 범위를 정확히 읽는다."""

from functools import lru_cache
import hashlib
import json
from pathlib import Path
import pickle

import numpy as np
import torch
from torch.utils.data import Dataset

from fedsec.model import PREPROCESSING
from fedsec.validation import positive_int

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@lru_cache(maxsize=2)
def _read_batch(path: str, size: int, modified: int):
    del size, modified
    raw = Path(path).read_bytes()
    batch = pickle.loads(raw, encoding="bytes")
    data = np.asarray(batch.get(b"data", batch.get("data")))
    labels = np.asarray(batch.get(b"labels", batch.get("labels")))
    if data.ndim != 2 or data.shape[1] != 3072 or labels.shape != (len(data),):
        raise ValueError("Expected CIFAR-10 data [N,3072] and labels [N]")
    if data.dtype != np.uint8 or not np.issubdtype(labels.dtype, np.integer) or np.any((labels < 0) | (labels > 9)):
        raise ValueError("Invalid CIFAR-10 pixels or labels")
    return data, labels, hashlib.sha256(raw).hexdigest()


class CifarRangeDataset(Dataset):
    def __init__(self, payload: dict, limit: int | None = None):
        if limit is not None:
            positive_int(limit, "sample limit")
        if not isinstance(payload, dict) or not payload.get("source_dir") or not payload.get("ranges"):
            raise ValueError("Distribute CIFAR-10 data before starting the experiment")
        root = Path(payload["source_dir"])
        root = (root if root.is_absolute() else PROJECT_ROOT / root).resolve()
        chunks, targets, records, seen = [], [], [], {}
        remaining = limit
        for segment in payload["ranges"]:
            path = (root / segment["file"]).resolve()
            if not path.is_relative_to(root):
                raise ValueError("Dataset file must stay inside source_dir")
            start = positive_int(segment["start"], "range start", 0)
            end = positive_int(segment["end"], "range end", 0)
            stat = path.stat()
            data, labels, digest = _read_batch(str(path), stat.st_size, stat.st_mtime_ns)
            if not 0 <= start < end <= len(data):
                raise ValueError("Dataset range is empty or out of bounds")
            for previous_start, previous_end in seen.setdefault(str(path), []):
                if start < previous_end and previous_start < end:
                    raise ValueError("Dataset ranges overlap")
            seen[str(path)].append((start, end))
            used_end = end if remaining is None else min(end, start + remaining)
            if used_end > start:
                chunks.append(data[start:used_end])
                targets.append(labels[start:used_end])
                records.append({"file": path.relative_to(root).as_posix(), "sha256": digest, "start": start, "end": used_end})
                if remaining is not None:
                    remaining -= used_end - start
        if not chunks:
            raise ValueError("Dataset is empty")
        self.images = np.concatenate(chunks).reshape(-1, 3, 32, 32)
        self.labels = np.concatenate(targets).astype(np.int64)
        self.fingerprint = hashlib.sha256(json.dumps(records, sort_keys=True).encode()).hexdigest()
        self.mean = torch.tensor(PREPROCESSING["mean"]).reshape(3, 1, 1)
        self.std = torch.tensor(PREPROCESSING["std"]).reshape(3, 1, 1)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        image = torch.from_numpy(self.images[index].copy()).float().div_(255)
        return (image - self.mean) / self.std, int(self.labels[index])


def load_datasets(manifest: dict, settings) -> tuple[dict[int, Dataset], Dataset]:
    """이미 결정된 데이터 배정표를 읽는다. 분할 정책/다운로드/네트워크 처리는 하지 않는다.

    manifest: source_dir, clients=[{client_id, ranges}], test_ranges=[{file,start,end}].
    학습·테스트 및 클라이언트끼리 겹치는 범위를 거부한다.
    """
    source = manifest["source_dir"]
    clients = manifest["clients"]
    if not clients:
        raise ValueError("Manifest contains no clients")
    ids = [positive_int(item["client_id"], "client_id", 0) for item in clients]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate client ID in manifest")
    root = Path(source)
    root = (root if root.is_absolute() else PROJECT_ROOT / root).resolve()
    seen = {}
    for ranges in [*[item["ranges"] for item in clients], manifest["test_ranges"]]:
        for segment in ranges:
            path = (root / segment["file"]).resolve()
            if not path.is_relative_to(root):
                raise ValueError("Dataset file must stay inside source_dir")
            start = positive_int(segment["start"], "range start", 0)
            end = positive_int(segment["end"], "range end")
            if start >= end:
                raise ValueError("Empty dataset range")
            if any(start < old_end and old_start < end for old_start, old_end in seen.get(path, [])):
                raise ValueError("Client/test dataset ranges overlap")
            seen.setdefault(path, []).append((start, end))
    datasets = {item["client_id"]: CifarRangeDataset({"source_dir": source, "ranges": item["ranges"]},
                                                    settings.max_train_samples) for item in clients}
    test = CifarRangeDataset({"source_dir": source, "ranges": manifest["test_ranges"]}, settings.max_test_samples)
    return datasets, test
