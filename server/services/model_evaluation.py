"""공통 fedsec 평가 함수를 기존 REST 응답 형식으로 연결한다."""

from common.data import CifarRangeDataset
from common.model import build_model, weights_hash
from common.training import TrainingSettings
from fedsec.data import dataset_fingerprint
from fedsec.training import evaluate_global_model as evaluate_core


def count_test_samples(dataset_range: dict) -> int:
    return sum(item["end"] - item["start"] for item in dataset_range.get("ranges", []))


def evaluate_global_model(global_model: dict, dataset_range: dict,
                          evaluation_config: dict, round_id: int) -> dict:
    settings = TrainingSettings.from_dict(evaluation_config)
    dataset = CifarRangeDataset(dataset_range, settings.max_test_samples)
    seed = evaluation_config.get("seed", 42)
    client_ids = evaluation_config.get("client_evaluation_ids", [])
    fixed_limit = evaluation_config.get("client_evaluation_max_samples", settings.max_test_samples)
    shared_pass = bool(client_ids) and fixed_limit == settings.max_test_samples
    matches = []

    def record_matches(batch):
        matches.extend(batch.tolist())

    model = build_model(global_model)
    result = evaluate_core(model, dataset, settings, seed=seed,
                           **({"record_matches": record_matches} if shared_pass else {}))
    identity = {
        "experiment_id": global_model.get("experiment_id"), "seed": seed, "round": round_id,
        "model_version": global_model.get("model_version"),
        "model_hash": weights_hash(global_model["weights"]) if "weights" in global_model else None,
        "split": dataset_range.get("split", "test"),
    }
    evaluation = {**identity, "accuracy": result.accuracy, "loss": result.loss, "num_test_samples": result.num_samples,
                  "correct_predictions": result.correct_predictions, "dataset_fingerprint": dataset.fingerprint,
                  "metrics": {"top1_accuracy": result.accuracy, "per_class": result.per_class}}
    if client_ids:
        if not shared_pass:
            settings = TrainingSettings.from_dict({**evaluation_config, "max_test_samples": fixed_limit})
            dataset = CifarRangeDataset(dataset_range, fixed_limit)
            evaluate_core(model, dataset, settings, seed=seed, record_matches=record_matches)
        evaluation["client_evaluations"] = _client_evaluations(client_ids, dataset, matches, identity)
    return evaluation


def _client_evaluations(client_ids: list[int], dataset, matches: list[bool], identity: dict) -> list[dict]:
    """Partition the ordered held-out set by registered ID, including nonparticipants."""
    if len(matches) != len(dataset):
        raise RuntimeError("Held-out match count does not match the dataset")
    size, extra = divmod(len(dataset), len(client_ids))
    rows = []
    start = 0
    for position, client_id in enumerate(sorted(client_ids)):
        count = size + (position < extra)
        end = start + count
        selection = {"parent_dataset_fingerprint": dataset.fingerprint, "start": start, "end": end}
        correct = sum(matches[start:end]) if count else None
        rows.append({
            **identity, "client_id": client_id, "model_role": "global",
            "dataset_selection": selection, "dataset_fingerprint": dataset_fingerprint([selection]) if count else None,
            "status": "evaluated" if count else "no_held_out_samples",
            "num_test_samples": count if count else None, "correct_predictions": correct,
            "accuracy": correct / count if count else None,
        })
        start = end
    return rows
