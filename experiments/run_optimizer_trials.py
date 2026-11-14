"""옵티마이저 변경(E27)과 후보 드롭아웃 재검증(E28)을 실행한다."""

import argparse
from copy import deepcopy
from pathlib import Path

from experiments.trial_configs import add_run_options, read_validation_config, run_configs
from fedsec.validation import dropout_probability


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description="Adam과 드롭아웃 차순위 후보를 검증합니다.")
    parser.add_argument("--base-config", type=Path, required=True, help="선택한 학습 일정의 SGD 설정")
    parser.add_argument("--dropout-base-config", type=Path, help="E28의 최종 후보 설정. 생략하면 --base-config를 사용합니다.")
    parser.add_argument("--runner-up-dropout", type=float, help="E28에서 재검증할 드롭아웃 확률")
    add_run_options(parser, ("E27", "E28"))
    args = parser.parse_args(argv)
    configs = []
    try:
        base, settings = read_validation_config(args.base_config)
        if args.trial != "E28":
            if settings.training.optimizer != "sgd":
                raise ValueError("E27 기준 설정의 옵티마이저는 SGD여야 합니다.")
            payload = deepcopy(base)
            payload["experiment_id"] = "E27_adam"
            payload["experiment"]["training"].update(optimizer="adam", learning_rate=0.001, momentum=0.0)
            configs.append(payload)
        if args.trial != "E27":
            if args.runner_up_dropout is None:
                raise ValueError("E28에 --runner-up-dropout을 지정하세요. E27만 실행하려면 --trial E27을 사용하세요.")
            probability = dropout_probability(args.runner_up_dropout)
            if args.dropout_base_config:
                base, settings = read_validation_config(args.dropout_base_config)
            if probability == settings.model_config.get("dropout", 0.0):
                raise ValueError("E28 드롭아웃은 최종 후보의 드롭아웃과 달라야 합니다.")
            payload = deepcopy(base)
            payload["experiment_id"] = "E28_dropout_runner_up"
            payload["experiment"]["model_config"] = {**settings.model_config, "dropout": probability}
            configs.append(payload)
    except (OSError, KeyError, TypeError, ValueError) as error:
        parser.error(str(error))
    run_configs(configs, args)


if __name__ == "__main__":
    main()
