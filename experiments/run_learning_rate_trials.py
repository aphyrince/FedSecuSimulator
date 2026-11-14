"""선택한 두 모델의 학습률 탐색 실험 E06~E13을 실행한다."""

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from fedsec.__main__ import main as run_model_trials


CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"
MODEL_CONFIGS = {
    "E03": "e03_smallcnn.json",
    "E04": "e04_cnn4.json",
    "E05": "e05_resnet18.json",
}
LEARNING_RATES = (0.01, 0.02, 0.05, 0.10)
TRIAL_IDS = tuple(f"E{number:02d}" for number in range(6, 14))


def main():
    parser = argparse.ArgumentParser(description="두 후보 모델의 학습률을 비교합니다. E03은 SmallCNN, E04는 4층 CNN, E05는 ResNet-18입니다.")
    parser.add_argument("--models", nargs=2, required=True, type=str.upper, choices=MODEL_CONFIGS,
                        metavar=("FIRST", "SECOND"), help="서로 다른 모델 두 개를 지정합니다. 예: E03 E04")
    parser.add_argument("--trial", type=str.upper, choices=TRIAL_IDS, help="지정한 실험 하나만 실행합니다.")
    parser.add_argument("--dry-run", action="store_true", help="학습 없이 실행할 실험과 학습률을 표시합니다.")
    parser.add_argument("--output-dir", type=Path, help="결과 저장 경로. 기본값은 results/standalone입니다.")
    args = parser.parse_args()
    if args.models[0] == args.models[1]:
        parser.error("서로 다른 모델 두 개를 지정하세요.")

    with TemporaryDirectory(prefix="fedsec-lr-trials-") as directory:
        run_args = []
        for first_trial, model_id in zip((6, 10), args.models):
            source = CONFIG_DIR / MODEL_CONFIGS[model_id]
            payload = json.loads(source.read_text(encoding="utf-8-sig"))
            payload["datasets"] = str((source.parent / payload["datasets"]).resolve())
            model_name = source.stem.split("_", 1)[1]
            for number, rate in enumerate(LEARNING_RATES, start=first_trial):
                trial_id = f"E{number:02d}"
                if args.trial and args.trial != trial_id:
                    continue
                rate_label = f"{rate:.2f}".replace(".", "p")
                experiment_id = f"{trial_id}_{model_name}_lr{rate_label}"
                payload["experiment_id"] = experiment_id
                payload["experiment"]["training"]["learning_rate"] = rate
                target = Path(directory) / f"{experiment_id}.json"
                target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
                run_args.extend(("--config", str(target)))

        if args.dry_run:
            run_args.append("--dry-run")
        if args.output_dir is not None:
            run_args.extend(("--output-dir", str(args.output_dir)))
        run_model_trials(run_args)


if __name__ == "__main__":
    main()
