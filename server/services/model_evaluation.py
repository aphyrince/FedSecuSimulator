"""공통 fedsec 평가 함수를 기존 REST 응답 형식으로 연결한다."""

from common.data import CifarRangeDataset
from common.model import build_model
from common.training import TrainingSettings
from fedsec.training import evaluate_global_model as evaluate_core


def count_test_samples(dataset_range: dict) -> int:
    return sum(item["end"] - item["start"] for item in dataset_range.get("ranges", []))


def evaluate_global_model(global_model: dict, dataset_range: dict,
                          evaluation_config: dict, round_id: int) -> dict:
    settings = TrainingSettings.from_dict(evaluation_config)
    dataset = CifarRangeDataset(dataset_range, settings.max_test_samples)
    result = evaluate_core(build_model(global_model), dataset, settings, seed=evaluation_config.get("seed", 42))
    return {"accuracy": result.accuracy, "loss": result.loss, "num_test_samples": result.num_samples,
            "correct_predictions": result.correct_predictions, "dataset_fingerprint": dataset.fingerprint,
            "metrics": {"top1_accuracy": result.accuracy}, "round": round_id}
