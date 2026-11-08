import csv
import json
from datetime import datetime, timezone
from pathlib import Path


ROUND_LOG_FIELDS = [
    "experiment_id",
    "round",
    "selected_clients",
    "accepted_clients",
    "delayed_clients",
    "malicious_selected",
    "malicious_accepted",
    "attack_enabled",
    "defense",
    "accuracy",
    "loss",
    "reason",
    "timestamp",
]


def save_experiment_outputs(
    results_dir: Path, config: dict, round_results: list[dict]
):
    experiment_dir = results_dir / config["experiment_id"]
    experiment_dir.mkdir(parents=True, exist_ok=True)
    log_path = experiment_dir / "rounds.csv"
    result_path = experiment_dir / "results.json"

    with log_path.open("w", newline="", encoding="utf-8-sig") as output:
        writer = csv.DictWriter(output, fieldnames=ROUND_LOG_FIELDS)
        writer.writeheader()
        for result in round_results:
            writer.writerow(
                {
                    key: (
                        ",".join(map(str, result[key]))
                        if isinstance(result[key], list)
                        else result[key]
                    )
                    for key in ROUND_LOG_FIELDS
                }
            )

    result_path.write_text(
        json.dumps(
            {
                "config": config,
                "round_results": round_results,
                "completed_at": datetime.now(timezone.utc).isoformat(),
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
