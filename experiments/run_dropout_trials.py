"""선택한 모델과 학습률로 드롭아웃 실험 E14~E17을 실행한다."""

import argparse
from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from fedsec.__main__ import _read_config, main as run_model_trials
from fedsec.validation import finite_number


DROPOUT_TRIALS = {"E14": 0.1, "E15": 0.2, "E16": 0.3, "E17": 0.4}


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description="같은 모델과 학습률에서 드롭아웃 확률을 비교합니다.")
    parser.add_argument("--base-config", required=True, type=Path, help="선택한 모델의 설정 또는 저장된 config.json")
    parser.add_argument("--learning-rate", type=float, help="학습률을 직접 지정합니다. 생략하면 기준 설정을 사용합니다.")
    parser.add_argument("--trial", type=str.upper, choices=DROPOUT_TRIALS, help="지정한 실험 하나만 실행합니다.")
    parser.add_argument("--dry-run", action="store_true", help="학습 없이 실험 설정을 확인합니다.")
    parser.add_argument("--output-dir", type=Path, help="결과 저장 경로. 기본값은 results/standalone입니다.")
    args = parser.parse_args(argv)
    try:
        base, settings = _read_config(args.base_config, None)
        if base["mode"] != "normal" or base["evaluation_split"] != "validation":
            raise ValueError("기준 설정은 정상 학습의 검증 실험이어야 합니다.")
        if settings.model_config.get("dropout", 0):
            raise ValueError("드롭아웃을 사용하지 않은 기준 설정을 지정하세요.")
        rate = settings.training.learning_rate if args.learning_rate is None else args.learning_rate
        finite_number(rate, "learning_rate", positive=True)
    except (OSError, KeyError, TypeError, ValueError) as error:
        parser.error(str(error))

    rate_label = str(rate).replace(".", "p").replace("+", "")
    base_id = base["experiment_id"]
    if len(base_id) > 50:
        parser.error("기준 experiment_id는 50자 이내로 지정하세요.")
    with TemporaryDirectory(prefix="fedsec-dropout-trials-") as directory:
        run_args = []
        selected = {args.trial: DROPOUT_TRIALS[args.trial]} if args.trial else DROPOUT_TRIALS
        for trial_id, probability in selected.items():
            probability_label = f"{probability:.1f}".replace(".", "p")
            experiment_id = f"{trial_id}_{base_id}_lr{rate_label}_dropout{probability_label}"
            payload = deepcopy(base)
            payload["experiment_id"] = experiment_id
            payload["experiment"]["model_config"] = {**settings.model_config, "dropout": probability}
            payload["experiment"]["training"] = {**settings.training.to_dict(), "learning_rate": rate}
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
