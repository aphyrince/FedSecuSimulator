"""E02~E37 독립 실험을 검증 결과에 따라 순서대로 실행한다."""

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "standalone" / "full_suite"
MODEL_CONFIGS = (
    "e02_smallcnn_baseline.json",
    "e03_smallcnn.json",
    "e04_cnn4.json",
    "e05_resnet18.json",
)


@dataclass(frozen=True)
class TrialResult:
    trial: str
    config_path: Path
    accuracy: float
    loss: float
    dropout: float


def read_json(path: Path) -> dict:
    with path.open(encoding="utf-8-sig") as file:
        return json.load(file)


def check_dataset_files():
    manifest = read_json(PROJECT_ROOT / "configs" / "cifar10_45k_5k.json")
    source = PROJECT_ROOT / manifest["source_dir"]
    files = {segment["file"] for client in manifest["clients"] for segment in client["ranges"]}
    files.update(segment["file"] for segment in manifest["test_ranges"])
    files.add(Path(manifest["test_ranges"][0]["file"]).with_name("test_batch").as_posix())
    missing = [str(source / name) for name in sorted(files) if not (source / name).is_file()]
    if missing:
        raise FileNotFoundError("CIFAR-10 파일이 없습니다.\n" + "\n".join(missing))


def run_trial(output_dir: Path, number: int, module: str, *options: str) -> TrialResult:
    trial = f"E{number:02d}"
    trial_dir = output_dir / trial
    mode = "attack" if number >= 35 else "normal"
    existing = list(trial_dir.glob(f"{trial}_*/{mode}/results.json"))
    if not existing:
        pending = list(trial_dir.glob(f"{trial}_*"))
        if len(pending) > 1:
            raise ValueError(f"{trial}의 미완료 실험 폴더가 여러 개 있습니다. {trial_dir}")
        if pending:
            config_path = pending[0] / "config.json"
            if not config_path.is_file():
                raise ValueError(f"{trial}의 저장된 설정이 없습니다. {pending[0]} 폴더를 다른 곳에 보관한 뒤 다시 실행하세요.")
            saved = read_json(config_path)
            if saved["experiment_id"] != pending[0].name or saved["mode"] != mode:
                raise ValueError(f"{trial}의 저장된 설정과 결과 폴더가 일치하지 않습니다. {config_path}")
            command = [sys.executable, "-m", "fedsec", "--config", str(config_path), "--resume"]
            print(f"[{trial}] 중단된 실험 이어서 실행", flush=True)
        else:
            command = [sys.executable, "-m", f"experiments.{module}", *options]
            print(f"[{trial}] 실행", flush=True)
        command.extend(("--output-dir", str(trial_dir)))
        subprocess.run(command, cwd=PROJECT_ROOT, check=True)
        existing = list(trial_dir.glob(f"{trial}_*/{mode}/results.json"))
    else:
        print(f"[{trial}] 저장된 결과 사용", flush=True)

    if len(existing) != 1:
        raise ValueError(f"{trial}의 결과 파일이 하나여야 합니다. {trial_dir}")
    report_path = existing[0]
    report = read_json(report_path)
    config_path = report_path.parents[1] / "config.json"
    config = read_json(config_path)
    rounds = report["round_results"]
    if (not report.get("checkpoint_verified") or not report_path.with_name("final_model.pt").is_file()
            or report.get("evaluation_split") != "validation"
            or len(rounds) != config["experiment"]["num_rounds"]
            or report["config"]["experiment_id"] != config["experiment_id"]
            or (number >= 29 and report.get("final_test", {}).get("num_test_samples") != 10000)):
        raise ValueError(f"{trial}의 결과가 완료된 실험과 일치하지 않습니다. {report_path}")
    accuracy, loss = rounds[-1]["accuracy"], rounds[-1]["loss"]
    if not 0 <= accuracy <= 1 or not math.isfinite(loss):
        raise ValueError(f"{trial}의 마지막 검증 정확도 또는 손실이 올바르지 않습니다. {report_path}")
    dropout = config["experiment"].get("model_config", {}).get("dropout", 0.0)
    return TrialResult(trial, config_path, accuracy, loss, dropout)


def rank(*candidates: TrialResult) -> list[TrialResult]:
    return sorted(candidates, key=lambda result: (-result.accuracy, result.loss, result.trial))


def select(description: str, *candidates: TrialResult) -> TrialResult:
    best = rank(*candidates)[0]
    print(f"{description} {best.trial} | 마지막 검증 정확도 {best.accuracy:.2%} | loss {best.loss:.4f}",
          flush=True)
    return best


def run_group(output_dir: Path, first: int, last: int, module: str, *options: str) -> list[TrialResult]:
    return [run_trial(output_dir, number, module, *options, "--trial", f"E{number:02d}")
            for number in range(first, last + 1)]


def run_suite(output_dir: Path):
    models = [run_trial(output_dir, number, "run_model_trials", "--config", str(PROJECT_ROOT / "configs" / name))
              for number, name in enumerate(MODEL_CONFIGS, start=2)]
    first, second = rank(*models[1:])[:2]
    print(f"학습률 비교 모델 {first.trial}, {second.trial}", flush=True)
    rates = run_group(output_dir, 6, 13, "run_learning_rate_trials", "--models", first.trial, second.trial)
    rate = select("학습률 선택", *rates)

    dropouts = run_group(output_dir, 14, 17, "run_dropout_trials", "--base-config", str(rate.config_path))
    dropout = select("드롭아웃 선택", rate, *dropouts)
    runner_up = rank(rate, *dropouts)[1]

    augmentations = run_group(output_dir, 18, 22, "run_augmentation_trials", "--base-config", str(dropout.config_path))
    augmentation = select("증강 선택", dropout, *augmentations)
    schedules = run_group(output_dir, 23, 26, "run_training_schedule_trials", "--base-config", str(augmentation.config_path))
    schedule = select("학습 일정 선택", augmentation, *schedules)
    adam, = run_group(output_dir, 27, 27, "run_optimizer_trials", "--base-config", str(schedule.config_path))
    optimizer = select("옵티마이저 선택", schedule, adam)
    candidate, = run_group(output_dir, 28, 28, "run_optimizer_trials", "--base-config", str(schedule.config_path),
                           "--dropout-base-config", str(optimizer.config_path), "--runner-up-dropout", str(runner_up.dropout))
    final = select("최종 설정 선택", optimizer, candidate)

    run_group(output_dir, 29, 37, "run_final_trials", "--base-config", str(final.config_path))
    print(f"E02~E37 전체 완료 | {output_dir}", flush=True)


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description="E02~E37 연합학습 독립 실험을 한 번에 실행합니다.")
    parser.add_argument("--dry-run", action="store_true", help="학습 없이 실행 순서와 결과 경로를 표시합니다.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="전체 실험 결과의 저장 경로")
    args = parser.parse_args(argv)
    output_dir = args.output_dir.resolve()
    if args.dry_run:
        print("E02~E05 모델 → E06~E13 학습률 → E14~E17 드롭아웃 → E18~E22 증강")
        print("→ E23~E26 학습 일정 → E27 Adam → E28 드롭아웃 재검증 → E29~E37 최종 평가")
        print(f"총 36개 실험 | 결과 {output_dir}")
        return

    try:
        check_dataset_files()
        run_suite(output_dir)
    except (OSError, KeyError, TypeError, ValueError) as error:
        parser.exit(2, f"오류: {error}\n")
    except subprocess.CalledProcessError as error:
        parser.exit(error.returncode, "실험이 중단됐습니다. 위 오류와 결과 폴더를 확인하세요.\n")
    except KeyboardInterrupt:
        parser.exit(130, "실험을 중단했습니다. 같은 명령을 실행하면 저장된 라운드부터 이어서 진행합니다.\n")


if __name__ == "__main__":
    main()
