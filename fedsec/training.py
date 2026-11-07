"""Local training is called explicitly by a user or the role-3 HTTP worker."""

from dataclasses import dataclass
import os
import random
import time
import numpy as np
import torch
from torch import nn
from .config import ModelConfig, RuntimeConfig, TrainingConfig, integer
from .contracts import ClientUpdate, RoundContext
from .data import make_loader
from .model import build_model
from .tensors import TensorState, clone_state, state_hash, state_norm, validate_state


def seed_everything(seed: int, *, deterministic: bool = True) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic
    torch.use_deterministic_algorithms(deterministic)


def resolve_device(runtime: RuntimeConfig) -> torch.device:
    name = runtime.device
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise ValueError("사용 가능한 CUDA 장치가 없습니다. runtime.device를 cpu로 바꾸세요.")
    return torch.device(name)


@dataclass
class LocalTrainingResult:
    update: ClientUpdate
    history: list[dict]


def train_local(global_state: TensorState, dataset, *, client_id: int, context: RoundContext,
                model_config: ModelConfig, training_config: TrainingConfig,
                runtime: RuntimeConfig, seed: int, on_epoch=None) -> LocalTrainingResult:
    """Use a fresh copy of W for every client; never mutate the supplied W.

    training_config contains all resolved hyperparameters; context contains only
    round identity/participation, with no HTTP request or competing settings.
    dataset is already assigned and normalized by the manager/role-3 adapter.
    Optimizer state is reset at each local call. num_samples is dataset size,
    never dataset size multiplied by the number of local epochs.
    """
    integer("client_id", client_id)
    if client_id not in context.selected_clients:
        raise ValueError("이번 라운드에 선정되지 않은 클라이언트입니다.")
    if len(dataset) < 1:
        raise ValueError("로컬 학습 데이터가 비어 있습니다.")
    base_hash = state_hash(global_state)
    local_seed = (seed + context.round_id * 100003 + client_id) % 2**32
    seed_everything(local_seed, deterministic=runtime.deterministic)
    torch.set_num_threads(runtime.torch_threads)
    device = resolve_device(runtime)
    model = build_model(model_config, global_state).to(device)
    loader = make_loader(dataset, batch_size=training_config.batch_size, shuffle=True,
                         seed=local_seed, num_workers=runtime.num_workers)
    if training_config.optimizer == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=training_config.learning_rate,
                                    momentum=training_config.momentum, weight_decay=training_config.weight_decay)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=training_config.learning_rate,
                                     weight_decay=training_config.weight_decay)
    criterion = nn.CrossEntropyLoss()
    history = []
    for epoch in range(1, training_config.local_epochs + 1):
        started = time.perf_counter()
        model.train()
        loss_sum = 0.0
        correct = count = 0
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            if not torch.isfinite(loss).item():
                raise ValueError("로컬 학습 loss가 NaN/Inf입니다.")
            loss.backward()
            optimizer.step()
            batch_count = labels.size(0)
            loss_sum += loss.item() * batch_count
            correct += (logits.argmax(dim=1) == labels).sum().item()
            count += batch_count
        if count != len(dataset):
            raise ValueError("실제 학습 표본 수가 배정 데이터 수와 다릅니다.")
        row = {"epoch": epoch, "loss": loss_sum / count, "accuracy": correct / count,
               "num_samples": count, "elapsed_seconds": time.perf_counter() - started}
        history.append(row)
        if on_epoch is not None:
            on_epoch(dict(row))
    final = clone_state(model.state_dict())
    delta = {name: final[name] - value.detach().cpu() for name, value in global_state.items()}
    validate_state(delta, global_state)
    update = ClientUpdate(context.experiment_id, client_id, context.round_id,
                          context.model_version, len(dataset), base_hash, delta)
    if on_epoch is not None:
        on_epoch({"phase": "local_complete", "update_norm": state_norm(delta)})
    return LocalTrainingResult(update, history)


def evaluate(model: nn.Module, loader, device: torch.device) -> dict:
    """Sample-weighted loss and accuracy in [0, 1]. Restores model mode."""
    previous_mode = model.training
    model.eval()
    loss_sum = 0.0
    correct = count = 0
    try:
        with torch.no_grad():
            for images, labels in loader:
                images, labels = images.to(device), labels.to(device)
                logits = model(images)
                loss = nn.functional.cross_entropy(logits, labels, reduction="sum")
                if not torch.isfinite(loss).item():
                    raise ValueError("평가 loss가 NaN/Inf입니다.")
                loss_sum += loss.item()
                correct += (logits.argmax(dim=1) == labels).sum().item()
                count += labels.size(0)
    finally:
        model.train(previous_mode)
    if not count:
        raise ValueError("평가 데이터가 비어 있습니다.")
    return {"loss": loss_sum / count, "accuracy": correct / count, "num_samples": count}
