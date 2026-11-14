"""데이터 증강까지 선택한 설정으로 집계 빈도 실험 E23~E26을 실행한다."""

import argparse
from copy import deepcopy
from pathlib import Path

from experiments.trial_configs import add_run_options, read_validation_config, run_configs


TRIALS = {"E23": (30, 1), "E24": (65, 1), "E25": (80, 1), "E26": (40, 2)}


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description="연합학습 라운드 수와 로컬 에포크를 비교합니다.")
    parser.add_argument("--base-config", type=Path, required=True, help="증강 단계에서 선택한 설정")
    add_run_options(parser, TRIALS)
    args = parser.parse_args(argv)
    try:
        base, _ = read_validation_config(args.base_config)
    except (OSError, KeyError, TypeError, ValueError) as error:
        parser.error(str(error))

    configs = []
    trials = (args.trial,) if args.trial else TRIALS
    for trial in trials:
        rounds, epochs = TRIALS[trial]
        payload = deepcopy(base)
        payload["experiment_id"] = f"{trial}_r{rounds}_ep{epochs}"
        payload["experiment"]["num_rounds"] = rounds
        payload["experiment"]["training"]["local_epochs"] = epochs
        configs.append(payload)
    run_configs(configs, args)


if __name__ == "__main__":
    main()
