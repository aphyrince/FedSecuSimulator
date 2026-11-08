"""팀원이 교체할 로컬 학습 인터페이스와 기본 가상 구현."""

from copy import deepcopy
from typing import Any, TypedDict


class LocalTrainingResult(TypedDict):
    """Flask 서버와 연동되는 로컬 학습 결과 형식."""

    local_model: dict[str, Any]
    model_delta: dict[str, Any] | float
    num_samples: int
    training_loss: float


class ExtractedModelUpdate(TypedDict):
    """학습 전후 전역 모델 차이로 추출한 업데이트."""

    values: dict[str, Any] | float
    update_norm: float


def count_dataset_samples(dataset_range: dict[str, Any]) -> int:
    """분배된 데이터 범위의 전체 표본 수를 계산한다."""
    return sum(
        max(0, int(segment.get("end", 0)) - int(segment.get("start", 0)))
        for segment in dataset_range.get("ranges", [])
    )


def train_local_model(
    global_model: dict[str, Any],
    dataset_range: dict[str, Any],
    training_config: dict[str, Any],
    round_id: int,
    client_id: int,
) -> LocalTrainingResult:
    """전역 모델과 로컬 데이터를 사용해 로컬 학습을 수행한다.

    현재 구현은 실제 학습을 대체하는 가상 결과다. 실제 팀은 이 함수의
    함수 시그니처와 LocalTrainingResult 형식을 유지한 채 본문만 교체한다.
    """
    num_samples = count_dataset_samples(dataset_range)
    local_model = deepcopy(global_model)
    local_model["model_version"] = round_id
    seed = int(training_config.get("seed", 42)) + client_id + round_id
    weights = local_model.get("weights")
    if isinstance(weights, dict):
        perturbation = round(((seed % 17) - 8) / 1000.0, 6)
        for name, values in weights.items():
            if isinstance(values, list) and values:
                if all(isinstance(value, (int, float)) for value in values):
                    weights[name] = [
                        round(float(value) + perturbation, 6) for value in values
                    ]

    model_delta = round(((seed % 17) - 8) / 100.0, 4)
    return {
        "local_model": local_model,
        "model_delta": model_delta,
        "num_samples": num_samples,
        "training_loss": 1.0,
    }


def _numeric_difference(before: Any, after: Any) -> Any:
    """JSON 직렬화 가능한 모델 파라미터의 숫자 차이를 계산한다."""
    if isinstance(before, dict) and isinstance(after, dict):
        keys = sorted(set(before) | set(after))
        return {
            key: _numeric_difference(before.get(key, 0), after.get(key, 0))
            for key in keys
        }
    if isinstance(before, list) and isinstance(after, list):
        return [
            _numeric_difference(before[index] if index < len(before) else 0, value)
            for index, value in enumerate(after)
        ]
    if isinstance(before, (int, float)) and isinstance(after, (int, float)):
        return after - before
    return None


def _square_values(value: Any) -> float:
    if isinstance(value, dict):
        return sum(_square_values(item) for item in value.values())
    if isinstance(value, list):
        return sum(_square_values(item) for item in value)
    if isinstance(value, (int, float)):
        return float(value * value)
    return 0.0


def extract_model_update(
    global_model_before_training: dict[str, Any],
    local_model_after_training: dict[str, Any],
) -> ExtractedModelUpdate:
    """학습 전후 모델 차이를 추출하여 클라이언트가 보관할 업데이트를 만든다.

    실제 팀이 모델 형태를 바꾸더라도 이 함수의 파라미터와 반환 형식만 유지한다.
    모델 차이 계산을 Tensor나 PyTorch 옵티마이저 업데이트로 교체할 수 있다.
    """
    before = global_model_before_training.get("weights", global_model_before_training)
    after = local_model_after_training.get("weights", local_model_after_training)
    values = _numeric_difference(before, after)
    return {
        "values": values,
        "update_norm": round(_square_values(values) ** 0.5, 8),
    }
