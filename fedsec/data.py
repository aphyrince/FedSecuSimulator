"""Dataset helpers for standalone execution; role 3 may supply its own Dataset."""

from collections import Counter
import hashlib
import random
import torch
from torch.utils.data import DataLoader, Subset
from .config import DataConfig, integer


def load_cifar10(config: DataConfig, *, train: bool, download: bool = False):
    from torchvision.datasets import CIFAR10
    from torchvision.transforms import Compose, Normalize, ToTensor
    from .config import project_path
    transform = Compose([ToTensor(), Normalize(config.normalization_mean, config.normalization_std)])
    return CIFAR10(root=str(project_path(config.root)), train=train, download=download, transform=transform)


def targets_hash(targets: list[int]) -> str:
    return hashlib.sha256(bytes(targets)).hexdigest()


def make_iid_partition(targets: list[int], *, num_clients: int, validation_size: int, seed: int) -> dict:
    """Class-stratified holdout and round-robin clients; divisibility is not required."""
    integer("num_clients", num_clients, 1)
    integer("validation_size", validation_size)
    if not targets or validation_size >= len(targets):
        raise ValueError("검증 데이터는 전체 데이터보다 작아야 합니다.")
    labels = sorted(set(targets))
    if labels != list(range(10)):
        raise ValueError("CIFAR-10 레이블 0~9가 필요합니다.")
    rng = random.Random(seed)
    clients = {str(i): [] for i in range(num_clients)}
    validation = []
    offset = 0
    for position, label in enumerate(labels):
        indices = [i for i, target in enumerate(targets) if target == label]
        rng.shuffle(indices)
        holdout = validation_size // len(labels) + (position < validation_size % len(labels))
        if holdout >= len(indices):
            raise ValueError("클래스별 학습 데이터가 남도록 validation_size를 줄이세요.")
        validation.extend(indices[:holdout])
        for index in indices[holdout:]:
            clients[str(offset % num_clients)].append(index)
            offset += 1
    rng.shuffle(validation)
    for indices in clients.values():
        rng.shuffle(indices)
    result = {"format_version": 1, "dataset": "CIFAR-10", "source_split": "official_train",
              "source_size": len(targets), "targets_sha256": targets_hash(targets),
              "seed": seed, "num_clients": num_clients, "client_indices": clients,
              "validation_indices": validation,
              "client_class_counts": {key: dict(Counter(targets[i] for i in ids)) for key, ids in clients.items()}}
    validate_partition(result, targets, num_clients=num_clients)
    return result


def validate_partition(partition: dict, targets: list[int], *, num_clients: int) -> None:
    if not isinstance(partition, dict) or partition.get("format_version") != 1:
        raise ValueError("분할 파일의 format_version은 1이어야 합니다.")
    if partition.get("dataset") != "CIFAR-10" or partition.get("source_split") != "official_train":
        raise ValueError("공식 학습 데이터 기준의 CIFAR-10 분할이 필요합니다.")
    if partition.get("source_size") != len(targets) or partition.get("targets_sha256") != targets_hash(targets):
        raise ValueError("분할 파일이 현재 원본 데이터와 다릅니다.")
    clients = partition.get("client_indices")
    if partition.get("num_clients") != num_clients or not isinstance(clients, dict) or set(clients) != {str(i) for i in range(num_clients)}:
        raise ValueError("분할 파일의 클라이언트 수/ID가 설정과 다릅니다.")
    validation = partition.get("validation_indices")
    if not isinstance(validation, list):
        raise ValueError("validation_indices 목록이 필요합니다.")
    groups = [validation]
    for indices in clients.values():
        if not isinstance(indices, list) or not indices:
            raise ValueError("클라이언트별 학습 인덱스가 비어 있습니다.")
        groups.append(indices)
    seen = set()
    for indices in groups:
        for index in indices:
            integer("dataset index", index)
            if index >= len(targets) or index in seen:
                raise ValueError("데이터 범위 초과 또는 학습/검증/클라이언트 간 중복입니다.")
            seen.add(index)
    if len(seen) != len(targets):
        raise ValueError("분할 파일은 원본 학습 데이터 전체를 한 번씩 배정해야 합니다.")


def make_loader(dataset, *, batch_size: int, shuffle: bool, seed: int, num_workers: int):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                      generator=torch.Generator().manual_seed(seed), drop_last=False)


def assigned_subset(dataset, indices: list[int], limit: int | None = None):
    selected = indices if limit is None else indices[:limit]
    if not selected:
        raise ValueError("배정된 데이터가 비어 있습니다.")
    return Subset(dataset, selected)
