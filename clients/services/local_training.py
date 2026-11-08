"""팀원이 교체할 로컬 학습 인터페이스와 기본 가상 구현."""

from copy import deepcopy
from typing import Any, TypedDict


class LocalTrainingResult(TypedDict):
    """Flask 서버와 연동되는 로컬 학습 결과 형식."""

    local_model: dict[str, Any]
    model_delta: dict[str, Any] | float
    num_samples: int
    training_loss: float


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
    model_delta = round(((seed % 17) - 8) / 100.0, 4)
    return {
        "local_model": local_model,
        "model_delta": model_delta,
        "num_samples": num_samples,
        "training_loss": 1.0,
    }
