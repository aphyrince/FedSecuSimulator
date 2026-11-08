"""팀원이 교체할 전역 모델 평가 인터페이스와 기본 가상 구현."""

import random
from typing import Any, TypedDict


class EvaluationResult(TypedDict):
    """모델 평가 API의 결과 형식."""

    accuracy: float
    loss: float
    num_test_samples: int
    correct_predictions: int
    metrics: dict[str, Any]


def count_test_samples(dataset_range: dict[str, Any]) -> int:
    """중앙 서버에 분배된 테스트 범위의 표본 수를 계산한다."""
    return sum(
        max(0, int(segment.get("end", 0)) - int(segment.get("start", 0)))
        for segment in dataset_range.get("ranges", [])
    )


def evaluate_global_model(
    global_model: dict[str, Any],
    dataset_range: dict[str, Any],
    evaluation_config: dict[str, Any],
    round_id: int,
) -> EvaluationResult:
    """전역 모델과 테스트 데이터로 분류 성능을 평가한다.

    현재 구현은 실제 데이터셋 모델 추론을 대체하는 가상 평가다.
    팀원이 실제 CIFAR-10 평가를 구현할 때 이 함수의 파라미터와 반환 키를 유지한다.
    """
    num_test_samples = count_test_samples(dataset_range)
    seed = int(evaluation_config.get("seed", 42)) + round_id
    rng = random.Random(seed)
    weights = global_model.get("weights", {})
    model_signal = sum(
        abs(value)
        for values in weights.values()
        if isinstance(values, list)
        for value in values
        if isinstance(value, (int, float))
    )
    accuracy = min(
        0.99,
        max(0.05, 0.45 + min(model_signal, 1.0) * 0.08 + round_id * 0.01),
    )
    accuracy = round(accuracy + rng.uniform(-0.01, 0.01), 4)
    accuracy = min(0.99, max(0.05, accuracy))
    correct_predictions = round(num_test_samples * accuracy)
    loss = round(max(0.05, 1.0 - accuracy), 4)
    return {
        "accuracy": accuracy,
        "loss": loss,
        "num_test_samples": num_test_samples,
        "correct_predictions": correct_predictions,
        "metrics": {
            "top1_accuracy": accuracy,
        },
    }
