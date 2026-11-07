"""Explicit commands: validate is always safe; simulate alone starts training."""

import argparse
import json
from .config import PROJECT_ROOT, load_config


def main(argv=None):
    parser = argparse.ArgumentParser(description="FedSecuSimulator 역할 1 알고리즘")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("validate", "download-data", "simulate"):
        command = commands.add_parser(name)
        command.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "experiment.json"))
        if name != "download-data":
            command.add_argument("--mode", choices=("normal", "attack", "compare"), default="normal")
        if name == "simulate":
            command.add_argument("--run-name")
            command.add_argument("--dry-run", action="store_true", help="설정/참여 일정만 출력; 다운로드/학습/파일 생성 없음")
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config)
        if args.command == "download-data":
            from .data import load_cifar10
            training = load_cifar10(config.data, train=True, download=True)
            test = load_cifar10(config.data, train=False, download=True)
            print(f"데이터 준비 완료: 학습 {len(training):,}장 / 테스트 {len(test):,}장")
        else:
            from .simulation import execution_plan, simulate
            if args.command == "validate" or args.dry_run:
                print(json.dumps(execution_plan(config, args.mode), ensure_ascii=False, indent=2))
                print("설정 검증 완료. 데이터 다운로드와 학습은 실행하지 않았습니다.")
            else:
                simulate(config, mode=args.mode, run_name=args.run_name)
    except (ValueError, OSError, RuntimeError, KeyError, TypeError) as error:
        parser.exit(2, f"오류: {error}\n")


if __name__ == "__main__":
    main()
