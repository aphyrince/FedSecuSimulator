"""선택한 드롭아웃 설정으로 데이터 증강 실험 E18~E22를 실행한다."""

import argparse
from copy import deepcopy
from pathlib import Path

from experiments.trial_configs import add_run_options, read_validation_config, run_configs


TRIALS = {
    "E18": "flip", "E19": "crop", "E20": "flip_crop",
    "E21": "flip_crop_jitter", "E22": "flip_crop_cutout",
}


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description="같은 모델과 드롭아웃에서 CIFAR-10 증강 효과를 비교합니다.")
    parser.add_argument("--base-config", type=Path, required=True, help="드롭아웃 단계에서 선택한 설정")
    add_run_options(parser, TRIALS)
    args = parser.parse_args(argv)
    try:
        base, settings = read_validation_config(args.base_config)
        if settings.training.augmentation != "none":
            raise ValueError("증강을 사용하지 않은 기준 설정을 지정하세요.")
    except (OSError, KeyError, TypeError, ValueError) as error:
        parser.error(str(error))

    configs = []
    trials = (args.trial,) if args.trial else TRIALS
    for trial in trials:
        payload = deepcopy(base)
        payload["experiment_id"] = f"{trial}_{TRIALS[trial]}"
        payload["experiment"]["training"]["augmentation"] = TRIALS[trial]
        configs.append(payload)
    run_configs(configs, args)


if __name__ == "__main__":
    main()
