"""학습이나 서버를 실행하지 않고 저장된 실제 실험 결과를 비교한다."""

import argparse
import json
from pathlib import Path
from fedsec.validation import finite_number


def compare_accuracy_results(normal: dict, attacked: dict) -> dict:
    """초기 모델·데이터·학습량·참여 조건이 같은지 검사한다.

    공격·방어 설정은 비교 조건으로 기록한다. 설정만으로 공격이나 방어의
    성공 여부를 판단하지 않는다.
    """
    normal_rounds, attack_rounds = normal.get("round_results", []), attacked.get("round_results", [])
    if not normal_rounds or len(normal_rounds) != len(attack_rounds):
        raise ValueError("Compare completed experiments with the same number of rounds")
    left = {item["round"]: item for item in normal_rounds}
    right = {item["round"]: item for item in attack_rounds}
    if len(left) != len(normal_rounds) or len(right) != len(attack_rounds) or set(left) != set(right):
        raise ValueError("Round numbers differ or are duplicated")
    if any(item.get("attack_enabled") for item in normal_rounds) or not any(item.get("attack_enabled") for item in attack_rounds):
        raise ValueError("Expected an attack-free baseline and an attack-enabled experiment")
    rows = []
    for round_id in sorted(left):
        baseline, attack = left[round_id], right[round_id]
        for key in ("initial_model_hash", "training_config", "training_datasets"):
            if not baseline.get(key) or baseline[key] != attack.get(key):
                raise ValueError(f"Cannot compare round {round_id}: different or missing {key}")
        if any(not item.get("fingerprint") for item in baseline["training_datasets"].values()):
            raise ValueError("Missing training dataset fingerprints")
        for key in ("selected_clients", "accepted_clients"):
            if not baseline.get(key) or sorted(baseline[key]) != sorted(attack.get(key, [])):
                raise ValueError(f"Cannot compare round {round_id}: {key} differ")
        left_eval, right_eval = baseline.get("evaluation", {}), attack.get("evaluation", {})
        fingerprint = left_eval.get("dataset_fingerprint")
        if not fingerprint or fingerprint != right_eval.get("dataset_fingerprint") or left_eval.get("num_test_samples", 0) <= 0 or left_eval.get("num_test_samples") != right_eval.get("num_test_samples"):
            raise ValueError("Evaluation datasets differ or lack real evaluation metadata")
        normal_accuracy = finite_number(baseline["accuracy"], "accuracy")
        attack_accuracy = finite_number(attack["accuracy"], "accuracy")
        if not 0 <= normal_accuracy <= 1 or not 0 <= attack_accuracy <= 1:
            raise ValueError("Accuracy must be in [0, 1]")
        rows.append({"round": round_id, "normal_accuracy": normal_accuracy, "attack_accuracy": attack_accuracy,
                     "accuracy_drop": normal_accuracy - attack_accuracy,
                     "accuracy_drop_percentage_points": (normal_accuracy - attack_accuracy) * 100,
                     "normal_loss": finite_number(baseline["loss"], "loss"),
                     "attack_loss": finite_number(attack["loss"], "loss"),
                     "attack_applied": attack.get("attack_applied", False),
                     "normal_defense": baseline.get("defense", "none") if baseline.get("defense_enabled") else "none",
                     "attack_defense": attack.get("defense", "none") if attack.get("defense_enabled") else "none"})
    return {"normal_experiment_id": normal.get("config", {}).get("experiment_id"),
            "attack_experiment_id": attacked.get("config", {}).get("experiment_id"),
            "rounds": rows, "final": rows[-1], "accuracy_unit": "fraction_0_to_1"}


def main():
    parser = argparse.ArgumentParser(description="정상·공격 실험 결과 비교 (학습 실행 없음)")
    parser.add_argument("normal_results", type=Path)
    parser.add_argument("attack_results", type=Path)
    args = parser.parse_args()
    try:
        baseline = json.loads(args.normal_results.read_text(encoding="utf-8-sig"))
        attacked = json.loads(args.attack_results.read_text(encoding="utf-8-sig"))
        print(json.dumps(compare_accuracy_results(baseline, attacked), ensure_ascii=False, indent=2, allow_nan=False))
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(2, f"오류: {error}\n")


if __name__ == "__main__":
    main()
