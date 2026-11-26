import csv
import json
import re
from datetime import datetime, timezone
from pathlib import Path


ROUND_LOG_FIELDS = [
    "experiment_id",
    "seed",
    "round",
    "start_model_version",
    "start_model_hash",
    "end_model_version",
    "end_model_hash",
    "selected_clients",
    "trained_clients",
    "accepted_clients",
    "unselected_clients",
    "excluded_clients",
    "selected_count",
    "trained_count",
    "accepted_count",
    "accepted_samples",
    "unselected_count",
    "excluded_count",
    "virtual_deadline_seconds",
    "virtual_elapsed_seconds",
    "virtual_last_arrival_seconds",
    "real_elapsed_seconds",
    "delayed_clients",
    "malicious_selected",
    "malicious_accepted",
    "malicious_ratio",
    "accepted_malicious_ratio",
    "attack_enabled",
    "attack_type",
    "attack_scale",
    "malicious_clients",
    "defense",
    "defense_enabled",
    "defense_type",
    "clip_norm",
    "fedavg_samples",
    "defense_aggregate_norm",
    "client_activity",
    "submission_attempts",
    "client_updates",
    "round_config",
    "model_updated",
    "accuracy",
    "loss",
    "evaluation",
    "reason",
    "timestamp",
]


def experiment_output_directory(results_dir: Path, config: dict) -> Path:
    experiment_id = config["experiment_id"]
    integrated_id = config.get("integrated_experiment_id")
    if integrated_id is None:
        return results_dir / experiment_id
    if not isinstance(integrated_id, str) or not re.fullmatch(
        r"[A-Za-z0-9_-]{1,100}", integrated_id
    ):
        raise ValueError("통합 실험 ID 형식이 올바르지 않습니다.")
    return results_dir / integrated_id / "cases" / experiment_id


def save_round_result(results_dir: Path, config: dict, result: dict) -> Path:
    """단일 라운드 결과를 실험 폴더의 JSON 파일로 즉시 기록한다."""
    experiment_dir = experiment_output_directory(results_dir, config)
    rounds_dir = experiment_dir / "rounds"
    rounds_dir.mkdir(parents=True, exist_ok=True)
    round_path = rounds_dir / f"round_{int(result['round']):03d}.json"
    round_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return round_path


def save_experiment_outputs(
    results_dir: Path, config: dict, round_results: list[dict], *, final_test: dict | None = None
):
    experiment_dir = experiment_output_directory(results_dir, config)
    experiment_dir.mkdir(parents=True, exist_ok=True)
    log_path = experiment_dir / "rounds.csv"
    result_path = experiment_dir / "results.json"
    if final_test is not None:
        (experiment_dir / "final_test.json").write_text(
            json.dumps(final_test, ensure_ascii=False, indent=2), encoding="utf-8",
        )

    with log_path.open("w", newline="", encoding="utf-8-sig") as output:
        writer = csv.DictWriter(output, fieldnames=ROUND_LOG_FIELDS)
        writer.writeheader()
        for result in round_results:
            writer.writerow(
                {
                    key: (
                        json.dumps(value, ensure_ascii=False)
                        if isinstance(value, (dict, list))
                        else value
                        if value is not None
                        else ""
                    )
                    for key in ROUND_LOG_FIELDS
                    for value in [result.get(key, "")]
                }
            )

    result_path.write_text(
        json.dumps(
            {
                "config": config,
                "round_results": round_results,
                "completed_at": datetime.now(timezone.utc).isoformat(),
                **({"final_test": final_test} if final_test is not None else {}),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    chart_path = experiment_dir / "accuracy.png"
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        rounds = [result["round"] for result in round_results]
        accuracies = [result["accuracy"] for result in round_results]
        plt.figure(figsize=(8, 4.5))
        plt.plot(rounds, accuracies, marker="o")
        plt.xlabel("Round")
        plt.ylabel("Accuracy")
        plt.title(f'{config["experiment_id"]} Accuracy')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(chart_path, dpi=150)
        plt.close()
    except (ImportError, OSError):
        chart_path = experiment_dir / "accuracy.csv"
        with chart_path.open("w", newline="", encoding="utf-8-sig") as output:
            writer = csv.writer(output)
            writer.writerow(["round", "accuracy"])
            writer.writerows(
                (result["round"], result["accuracy"]) for result in round_results
            )
    return log_path, result_path, chart_path
