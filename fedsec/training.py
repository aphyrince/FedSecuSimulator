"""1번 담당: Dataset와 모델만으로 실행하는 로컬 학습/평가. Flask 및 파일 저장 없음."""

from copy import deepcopy
import time
from typing import Callable
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset

from fedsec.config import TrainingSettings, setup_runtime
from fedsec.contracts import LocalTrainingResult, EvaluationResult
from fedsec.tensors import copy_state, extract_update
from fedsec.validation import positive_int


def _limited(dataset: Dataset, limit: int | None) -> Dataset:
    if len(dataset) == 0:
        raise ValueError("Dataset is empty")
    return Subset(dataset, range(limit)) if limit is not None and len(dataset) > limit else dataset


def train_local_model(global_model: nn.Module, dataset: Dataset, settings: TrainingSettings,
                      *, seed: int = 42, round_id: int = 1, client_id: int = 0,
                      check_cancelled: Callable[[], None] | None = None) -> LocalTrainingResult:
    """초기 모델을 복사해서 학습한다. 매 라운드 optimizer 초기화, 표본 수는 epoch와 무관.

    check_cancelled는 배치마다 호출되는 선택적 콜백이다. 네트워크의 제출 기한은
    어댑터에서 이 콜백으로 검사하며 핵심 학습 함수는 HTTP 기한 형식을 알지 않는다.
    """
    positive_int(round_id, "round_id")
    positive_int(client_id, "client_id", 0)
    seed = (positive_int(seed, "seed", 0) + round_id * 100003 + client_id) % 2**32
    device = setup_runtime(settings, seed)
    dataset = _limited(dataset, settings.max_train_samples)
    initial = copy_state(global_model.state_dict())
    model = deepcopy(global_model).to(device)
    loader = DataLoader(dataset, batch_size=settings.batch_size, shuffle=True,
                        num_workers=settings.num_workers, drop_last=False,
                        generator=torch.Generator().manual_seed(seed))
    if settings.optimizer == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=settings.learning_rate,
                                    momentum=settings.momentum, weight_decay=settings.weight_decay)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=settings.learning_rate, weight_decay=settings.weight_decay)
    history = []
    for epoch in range(1, settings.local_epochs + 1):
        started = time.perf_counter()
        loss_sum, correct, count = 0.0, 0, 0
        model.train()
        for images, labels in loader:
            if check_cancelled:
                check_cancelled()
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = nn.functional.cross_entropy(logits, labels)
            if not torch.isfinite(loss).item():
                raise ValueError("Training loss is NaN/Infinity")
            loss.backward()
            optimizer.step()
            count += len(labels)
            loss_sum += loss.item() * len(labels)
            correct += (logits.argmax(1) == labels).sum().item()
        history.append({"epoch": epoch, "loss": loss_sum / count, "accuracy": correct / count,
                        "elapsed_seconds": time.perf_counter() - started})
    final = copy_state(model.state_dict(), initial)
    return LocalTrainingResult(final, extract_update(initial, final), len(dataset), history)


def evaluate_global_model(global_model: nn.Module, dataset: Dataset, settings: TrainingSettings,
                          *, seed: int = 42) -> EvaluationResult:
    device = setup_runtime(settings, seed)
    dataset = _limited(dataset, settings.max_test_samples)
    loader = DataLoader(dataset, batch_size=settings.batch_size, shuffle=False, num_workers=settings.num_workers)
    model = deepcopy(global_model).to(device)
    model.eval()
    correct, count, loss_sum = 0, 0, 0.0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss = nn.functional.cross_entropy(logits, labels, reduction="sum")
            if not torch.isfinite(loss).item():
                raise ValueError("Evaluation loss is NaN/Infinity")
            loss_sum += loss.item()
            correct += (logits.argmax(1) == labels).sum().item()
            count += len(labels)
    return EvaluationResult(correct / count, loss_sum / count, count, correct)
