"""이전 실험 설정을 읽고 새 조건을 공통 CLI에 전달한다."""

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from fedsec.__main__ import _read_config, main as run_model_trials


def read_validation_config(path: Path):
    payload, settings = _read_config(path, None)
    if payload["mode"] != "normal" or payload["evaluation_split"] != "validation":
        raise ValueError("기준으로 정상 학습의 검증 실험 config.json을 지정하세요.")
    manifest = payload["datasets"]
    if "final_test_ranges" in manifest:
        raise ValueError("최종 테스트 전의 검증 실험 설정을 지정하세요.")
    return payload, settings


def add_run_options(parser: argparse.ArgumentParser, trial_ids):
    parser.add_argument("--trial", type=str.upper, choices=trial_ids, help="지정한 실험 하나를 실행합니다.")
    parser.add_argument("--dry-run", action="store_true", help="학습 없이 실험 설정을 확인합니다.")
    parser.add_argument("--output-dir", type=Path, default=Path("results/standalone"), help="결과 저장 경로")


def run_configs(configs: list[dict], args: argparse.Namespace):
    with TemporaryDirectory(prefix="fedsec-trials-") as directory:
        run_args = ["--output-dir", str(args.output_dir)]
        if args.dry_run:
            run_args.append("--dry-run")
        for index, config in enumerate(configs):
            target = Path(directory) / f"trial_{index:02d}.json"
            target.write_text(json.dumps(config, ensure_ascii=False, allow_nan=False), encoding="utf-8")
            run_args.extend(("--config", str(target)))
        run_model_trials(run_args)
