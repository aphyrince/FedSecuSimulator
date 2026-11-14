"""E29~E37의 시드별 기준·최적 설정·공격 비교와 최종 테스트를 실행한다."""

import argparse
from copy import deepcopy
from pathlib import Path, PurePosixPath

from experiments.trial_configs import add_run_options, read_validation_config, run_configs
from fedsec.config import ExperimentSettings


BASELINE_CONFIG = Path(__file__).resolve().parents[1] / "configs/e02_smallcnn_baseline.json"
SEEDS = (101, 202, 303)
GROUPS = ((29, "baseline", "normal"), (32, "selected", "normal"), (35, "selected", "attack"))
TRIAL_IDS = tuple(f"E{number}" for number in range(29, 38))


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description="확정한 모델 설정을 세 시드로 다시 학습하고 공식 테스트셋을 평가합니다.")
    parser.add_argument("--base-config", type=Path, help="검증에서 선택한 최종 모델 설정")
    add_run_options(parser, TRIAL_IDS)
    args = parser.parse_args(argv)
    configs = []
    try:
        baseline, _ = read_validation_config(BASELINE_CONFIG)
        bases = {"baseline": baseline}
        if args.trial not in TRIAL_IDS[:3]:
            if args.base_config is None:
                raise ValueError("E32~E37에는 확정한 --base-config가 필요합니다.")
            selected, _ = read_validation_config(args.base_config)
            if selected["datasets"] != baseline["datasets"]:
                raise ValueError("최종 후보와 E02의 CIFAR-10 학습·검증 배정이 같아야 합니다.")
            bases["selected"] = selected

        test_file = PurePosixPath(baseline["datasets"]["test_ranges"][0]["file"]).with_name("test_batch")
        for start, group, mode in GROUPS:
            for number, seed in enumerate(SEEDS, start=start):
                trial = f"E{number}"
                if args.trial and args.trial != trial:
                    continue
                payload = deepcopy(bases[group])
                payload["experiment_id"] = f"{trial}_{group}_{mode}_seed{seed}"
                payload["mode"] = mode
                payload["datasets"]["final_test_ranges"] = [{"file": test_file.as_posix(), "start": 0, "end": 10000}]
                options = payload["experiment"]
                options["seed"] = seed
                options["training"].update(max_train_samples=None, max_test_samples=None)
                if mode == "attack" and not options.get("malicious_clients"):
                    options["malicious_clients"] = list(ExperimentSettings().malicious_clients)
                if mode == "attack" and (len(options["malicious_clients"]) != 2 or not set(options["malicious_clients"]) <= {
                    client["client_id"] for client in payload["datasets"]["clients"]
                }):
                    raise ValueError("최종 공격 비교에는 배정표에 있는 클라이언트 두 개를 지정하세요.")
                configs.append(payload)
    except (OSError, IndexError, KeyError, TypeError, ValueError) as error:
        parser.error(str(error))
    run_configs(configs, args)


if __name__ == "__main__":
    main()
