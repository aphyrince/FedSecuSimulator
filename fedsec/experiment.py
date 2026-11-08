"""1번 담당 독립 실행: 가상 클라이언트를 순차 학습한다. HTTP/프로세스/파일 저장 없음."""

from copy import deepcopy
from dataclasses import dataclass
import hashlib
from typing import Callable
from torch.utils.data import Dataset, Subset

from fedsec.aggregation import aggregate_fedavg
from fedsec.attack import collude_updates
from fedsec.comparison import compare_accuracy_results
from fedsec.config import ExperimentSettings
from fedsec.contracts import ClientUpdate, TensorState
from fedsec.model import create_model
from fedsec.tensors import copy_state, state_hash
from fedsec.training import train_local_model, evaluate_global_model
from fedsec.validation import positive_int


@dataclass(frozen=True)
class ExperimentResult:
    config: dict
    round_results: list[dict]
    final_weights: TensorState

    def report(self) -> dict:
        """JSON 보고서에는 Tensor를 넣지 않는다. final_weights는 별도 체크포인트용이다."""
        return {"config": deepcopy(self.config), "round_results": deepcopy(self.round_results)}


@dataclass(frozen=True)
class ComparisonResult:
    normal: ExperimentResult
    attacked: ExperimentResult
    comparison: dict


def _dataset_view(dataset: Dataset, limit: int | None) -> Dataset:
    fingerprint = getattr(dataset, "fingerprint", None)
    if not isinstance(fingerprint, str) or not fingerprint or not len(dataset):
        raise ValueError("Each dataset requires samples and a stable fingerprint")
    if limit is not None and len(dataset) > limit:
        view = Subset(dataset, range(limit))
        view.fingerprint = hashlib.sha256(f"{fingerprint}:first:{limit}".encode()).hexdigest()
        return view
    return dataset


def _datasets(config, client_datasets, test_dataset):
    if not client_datasets:
        raise ValueError("At least one client is required")
    for client_id in client_datasets:
        positive_int(client_id, "client_id", 0)
    clients = {key: _dataset_view(client_datasets[key], config.training.max_train_samples)
               for key in sorted(client_datasets)}
    return clients, _dataset_view(test_dataset, config.training.max_test_samples)


def run_federated_learning(config: ExperimentSettings, client_datasets: dict[int, Dataset],
                           test_dataset: Dataset, *, attack: bool = False,
                           experiment_id: str = "standalone", initial_weights: TensorState | None = None,
                           on_round: Callable[[dict], None] | None = None) -> ExperimentResult:
    """같은 전역 모델로 모든 클라이언트 학습 → 선택적 협력 공격 → FedAvg → 평가를 반복한다.

    매 라운드 모든 클라이언트가 참여한다. 데이터 분할은 호출자가 제공한다.
    실제 학습은 이 함수를 호출할 때만 시작한다.
    """
    config = deepcopy(config)
    clients, test_dataset = _datasets(config, client_datasets, test_dataset)
    if attack and (len(config.malicious_clients) < 2 or not set(config.malicious_clients) <= clients.keys()):
        raise ValueError("Attack requires at least two configured clients present in the datasets")
    model = create_model(config.seed, config.model_config)
    if initial_weights is not None:
        model.load_state_dict(copy_state(initial_weights, model.state_dict()))
    initial_hash = state_hash(model.state_dict())
    records = []
    for round_id in range(1, config.num_rounds + 1):
        updates, training_metrics = [], {}
        for client_id, dataset in clients.items():
            trained = train_local_model(model, dataset, config.training, seed=config.seed,
                                        round_id=round_id, client_id=client_id)
            if trained.num_samples != len(dataset):
                raise ValueError("Local training sample count does not match the assigned dataset")
            updates.append(ClientUpdate(client_id, trained.num_samples, trained.delta))
            training_metrics[str(client_id)] = trained.history
        if attack:
            # 모든 정상 delta를 모은 뒤 변조한다. 변조한 값을 평균에 다시 섞지 않는다.
            clean = [item.delta for item in updates if item.client_id in config.malicious_clients]
            malicious_delta = collude_updates(clean, config.attack_scale)
            updates = [ClientUpdate(item.client_id, item.num_samples, copy_state(malicious_delta))
                       if item.client_id in config.malicious_clients else item for item in updates]
        model.load_state_dict(aggregate_fedavg(copy_state(model.state_dict()), updates))
        evaluation = evaluate_global_model(model, test_dataset, config.training, seed=config.seed)
        record = {"experiment_id": experiment_id, "round": round_id,
                  "initial_model_hash": initial_hash, "training_config": config.training.to_dict(),
                  "training_datasets": {str(key): {"fingerprint": dataset.fingerprint, "num_samples": len(dataset)}
                                        for key, dataset in clients.items()},
                  "selected_clients": list(clients), "accepted_clients": list(clients),
                  "accuracy": evaluation.accuracy, "loss": evaluation.loss,
                  "evaluation": {"dataset_fingerprint": test_dataset.fingerprint,
                                 "num_test_samples": evaluation.num_samples},
                  "attack_enabled": attack, "attack_applied": attack,
                  "attack_scale": config.attack_scale, "malicious_clients": list(config.malicious_clients) if attack else [],
                  "defense_enabled": False, "defense": "none", "training_metrics": training_metrics}
        records.append(record)
        if on_round:
            on_round(deepcopy(record))
    return ExperimentResult({**config.to_dict(), "experiment_id": experiment_id, "attack_enabled": attack},
                            records, copy_state(model.state_dict()))


def run_comparison_experiment(config: ExperimentSettings, client_datasets: dict[int, Dataset],
                              test_dataset: Dataset, *, experiment_id: str = "comparison",
                              on_round: Callable[[dict], None] | None = None) -> ComparisonResult:
    """네 담당의 통합 진입 함수: 같은 초기 모델로 정상 FL → 공격 FL → 정확도 비교."""
    config = deepcopy(config)
    clients, test_dataset = _datasets(config, client_datasets, test_dataset)
    if len(config.malicious_clients) < 2 or not set(config.malicious_clients) <= clients.keys():
        raise ValueError("Comparison requires at least two configured malicious clients")
    initial = copy_state(create_model(config.seed, config.model_config).state_dict())
    options = {"initial_weights": initial, "on_round": on_round}
    normal = run_federated_learning(config, clients, test_dataset, attack=False,
                                    experiment_id=f"{experiment_id}_normal", **options)
    attacked = run_federated_learning(config, clients, test_dataset, attack=True,
                                      experiment_id=f"{experiment_id}_attack", **options)
    return ComparisonResult(normal, attacked, compare_accuracy_results(normal.report(), attacked.report()))
