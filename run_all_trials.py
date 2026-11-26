"""E02~E37 독립 실험 또는 E03·E04의 모델별 설정 탐색을 실행한다."""

import argparse
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory

from experiments.cnn_report import REPORT_NAME, load_cnn_workbook, save_cnn_workbook
from experiments.trial_candidates import (
    ADAM_TRAINING,
    AUGMENTATION_TRIALS,
    DROPOUT_TRIALS,
    FINAL_SEEDS,
    FINAL_TEST_SIZE,
    LEARNING_RATES,
    SCHEDULE_TRIALS,
    add_official_test,
)


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "standalone" / "full_suite"
CNN_OUTPUT_DIR = DEFAULT_OUTPUT_DIR.with_name("cnn_only")
MODEL_CONFIGS = (
    "e02_smallcnn_baseline.json",
    "e03_smallcnn.json",
    "e04_cnn4.json",
    "e05_resnet18.json",
)
CNN_MODELS = (
    ("e03_smallcnn.json", "small_cnn_v1", 6),
    ("e04_cnn4.json", "cnn4_cifar10_v1", 10),
)
STAGE_LABELS = {
    "learning_rate": "학습률",
    "dropout": "드롭아웃",
    "augmentation": "데이터 증강",
    "schedule": "학습 일정",
    "optimizer": "옵티마이저",
}


@dataclass(frozen=True)
class TrialResult:
    trial: str
    config_path: Path
    accuracy: float
    loss: float
    dropout: float


@dataclass(frozen=True)
class FailedTrial:
    trial: str
    config_path: Path
    failure: dict


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
    return read_trial_result(trial, existing[0], final_test=number >= 29)


def read_trial_result(trial: str, report_path: Path, *, final_test: bool = False) -> TrialResult:
    report = read_json(report_path)
    config_path = report_path.parents[1] / "config.json"
    config = read_json(config_path)
    rounds = report["round_results"]
    if (not report.get("checkpoint_verified") or not report_path.with_name("final_model.pt").is_file()
            or report.get("evaluation_split") != "validation"
            or len(rounds) != config["experiment"]["num_rounds"]
            or report["config"]["experiment_id"] != config["experiment_id"]
            or (final_test and report.get("final_test", {}).get("num_test_samples") != FINAL_TEST_SIZE)):
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


def cnn_model_plans(device: str | None = None):
    plans = []
    for filename, architecture, first_rate in CNN_MODELS:
        source = PROJECT_ROOT / "configs" / filename
        base = read_json(source)
        base["datasets"] = read_json(source.parent / base["datasets"])
        options = base["experiment"]
        model, training = options["model_config"], options["training"]
        if device is not None:
            training["device"] = device
        if (base["mode"] != "normal" or base["evaluation_split"] != "validation"
                or "final_test_ranges" in base["datasets"] or model["architecture"] != architecture
                or model.get("dropout", 0.0) != 0.0 or training.get("augmentation", "none") != "none"
                or training["optimizer"] != "sgd"):
            raise ValueError(f"CNN 기본 설정은 드롭아웃과 증강이 없는 정상 SGD 검증 실험이어야 합니다. {source}")
        model.setdefault("dropout", 0.0)
        training.setdefault("augmentation", "none")
        if plans:
            first = plans[0][0]
            shared = {key: value for key, value in options.items() if key != "model_config"}
            first_shared = {key: value for key, value in first["experiment"].items() if key != "model_config"}
            if shared != first_shared or base["datasets"] != first["datasets"]:
                raise ValueError("E03과 E04의 기본 학습 조건, 시드와 데이터 배정은 같아야 합니다.")
        stages = {
            "learning_rate": {
                f"E{number:02d}_lr{rate:.2f}".replace(".", "p"): {"training": {"learning_rate": rate}}
                for number, rate in enumerate(LEARNING_RATES, start=first_rate)
            },
            "dropout": {
                f"{trial}_dropout{probability:.1f}".replace(".", "p"): {"model_config": {"dropout": probability}}
                for trial, probability in DROPOUT_TRIALS.items()
            },
            "augmentation": {
                f"{trial}_{augmentation}": {"training": {"augmentation": augmentation}}
                for trial, augmentation in AUGMENTATION_TRIALS.items()
            },
            "schedule": {
                f"{trial}_r{rounds}_ep{epochs}": {"num_rounds": rounds, "training": {"local_epochs": epochs}}
                for trial, (rounds, epochs) in SCHEDULE_TRIALS.items()
            },
            "optimizer": {"E27_adam": {"training": ADAM_TRAINING}},
        }
        plans.append((base, stages))
    return plans


def cnn_candidate_config(base: dict, experiment_id: str, overrides: dict) -> dict:
    config = deepcopy(base)
    config["experiment_id"] = experiment_id
    options = config["experiment"]
    for key, value in overrides.items():
        if key in ("training", "model_config"):
            options[key].update(value)
        else:
            options[key] = value
    return config


def cnn_final_configs(base: dict):
    for number, seed in enumerate(FINAL_SEEDS, start=32):
        config = cnn_candidate_config(base, f"E{number}_selected_normal_seed{seed}", {
            "seed": seed, "training": {"max_train_samples": None, "max_test_samples": None},
        })
        add_official_test(config["datasets"])
        yield config


def cnn_report_path(model_dir: Path, experiment_id: str) -> Path:
    return model_dir / experiment_id / "normal" / "results.json"


def print_cnn_plan(output_dir: Path, plans):
    base = plans[0][0]
    manifest = base["datasets"]
    train_count = sum(item["end"] - item["start"] for client in manifest["clients"] for item in client["ranges"])
    validation_count = sum(item["end"] - item["start"] for item in manifest["test_ranges"])
    print(f"CNN 전용 | {output_dir}")
    print(f"학습·평가 장치 {base['experiment']['training']['device']} | 최종 엑셀 {output_dir / REPORT_NAME}")
    print(f"공통 배정 {len(manifest['clients'])}개 클라이언트 | 학습 {train_count:,}장 | 검증 {validation_count:,}장"
          f" | 탐색 시드 {base['experiment']['seed']}")
    print("마지막 라운드 검증 정확도 내림차순, 손실 오름차순, 실험 ID 오름차순으로 모델별 선택")
    for base, stages in plans:
        model_id = base["experiment_id"]
        model_dir = output_dir / model_id
        options = base["experiment"]
        training = options["training"]
        print(f"\n{model_id} | {options['model_config']['architecture']} | 요약 {model_dir / 'summary.json'}")
        print(f"  학습률 기준 {training['learning_rate']:g} | 드롭아웃 0 | 증강 none | {options['num_rounds']}라운드"
              f" × 로컬 {training['local_epochs']} epoch | SGD | {cnn_report_path(model_dir, model_id)}")
        for stage, candidates in stages.items():
            baseline = "기준 설정 포함" if stage == "learning_rate" else "직전 단계의 선택 설정 포함"
            print(f"  {STAGE_LABELS[stage]} | {baseline}")
            if stage == "augmentation":
                print("    무증강 기준점 재사용 | 선택 학습률·드롭아웃·라운드 수 고정")
            for experiment_id, overrides in candidates.items():
                settings = json.dumps(overrides, ensure_ascii=False)
                print(f"    {experiment_id} | 변경 {settings} | {cnn_report_path(model_dir, experiment_id)}")
    print(f"\n모델별 최종 설정 확정 후 정상 학습 및 공식 테스트 | CIFAR-10 test_batch {FINAL_TEST_SIZE:,}장")
    for base, _ in plans:
        model_dir = output_dir / base["experiment_id"]
        for config in cnn_final_configs(base):
            print(f"  {base['experiment_id']} | 시드 {config['experiment']['seed']} | "
                  f"{cnn_report_path(model_dir, config['experiment_id'])}")


def run_cnn_trial(model_dir: Path, config: dict) -> TrialResult | FailedTrial:
    experiment_id = config["experiment_id"]
    with TemporaryDirectory(prefix="fedsec-cnn-trial-") as directory:
        config_path = Path(directory) / "config.json"
        write_json(config_path, config)
        command = [sys.executable, "-m", "fedsec", "--config", str(config_path),
                   "--output-dir", str(model_dir), "--resume", "--record-trial-failure"]
        subprocess.run(command, cwd=PROJECT_ROOT, check=True)
    report_path = cnn_report_path(model_dir, experiment_id)
    failure_path = report_path.with_name("failure.json")
    if failure_path.is_file() and not report_path.is_file():
        failure = read_json(failure_path)
        if failure["experiment_id"] != experiment_id:
            raise ValueError(f"실패 기록의 실험 ID가 일치하지 않습니다. {failure_path}")
        print(f"[{experiment_id}] 건너뜀 | {failure['reason']}", flush=True)
        return FailedTrial(experiment_id, report_path.parents[1] / "config.json", failure)
    return read_trial_result(experiment_id, report_path, final_test="final_test_ranges" in config["datasets"])


def write_json(path: Path, data: dict):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_cnn_summary(model_dir: Path, summary: dict, filename: str = "summary.json"):
    with TemporaryDirectory(prefix=".summary-", dir=model_dir) as directory:
        pending = Path(directory) / filename
        write_json(pending, summary)
        pending.replace(model_dir / filename)


def cnn_summary_entry(model_dir: Path, result: TrialResult | FailedTrial) -> dict:
    entry = {
        "experiment_id": result.trial,
        "config_path": result.config_path.relative_to(model_dir).as_posix(),
        "settings": read_json(result.config_path)["experiment"],
    }
    if isinstance(result, FailedTrial):
        return {**entry, **result.failure, "status": "failed", "validation_accuracy": None, "validation_loss": None}
    return {**entry, "validation_accuracy": result.accuracy, "validation_loss": result.loss}


def tune_cnn_models(output_dir: Path, plans):
    selected_models = {}
    for base, stages in plans:
        model_id = base["experiment_id"]
        model_dir = output_dir / model_id
        selected = run_cnn_trial(model_dir, base)
        selected_config = base
        summary = {"model": model_id, "stages": {}, "official_tests": []}
        for stage, candidates in stages.items():
            configs = {selected.trial: selected_config}
            configs.update(
                (trial, cnn_candidate_config(selected_config, trial, overrides))
                for trial, overrides in candidates.items()
            )
            results = [selected]
            results.extend(run_cnn_trial(model_dir, configs[trial]) for trial in candidates)
            comparison = {"candidates": [cnn_summary_entry(model_dir, result) for result in results]}
            summary["stages"][stage] = comparison
            completed = [result for result in results if isinstance(result, TrialResult)]
            if not completed:
                comparison["selected"] = None
                write_cnn_summary(model_dir, summary)
                raise ValueError(f"[{model_id}] {STAGE_LABELS[stage]} 단계에서 완료된 후보가 없습니다. {model_dir}")
            selected = select(f"[{model_id}] {STAGE_LABELS[stage]} 선택", *completed)
            selected_config = configs[selected.trial]
            comparison["selected"] = selected.trial
            if stage == "augmentation":
                baseline = results[0]  # 이 모델에서 선택한 무증강 드롭아웃 실험
                comparison["baseline_experiment_id"] = baseline.trial
                for entry in comparison["candidates"]:
                    accuracy = entry["validation_accuracy"]
                    entry["accuracy_delta_pp"] = None if accuracy is None else 100 * (accuracy - baseline.accuracy)
            write_cnn_summary(model_dir, summary)
        summary["final_selection"] = cnn_summary_entry(model_dir, selected)
        write_cnn_summary(model_dir, summary)
        selected_models[model_id] = {"config": selected_config, "validation_summary": summary}
    return selected_models


def run_cnn_suite(output_dir: Path, plans):
    plan_hash = hashlib.sha256(json.dumps(plans, sort_keys=True).encode()).hexdigest()
    selection_path = output_dir / "frozen_selections.json"
    if selection_path.exists():
        frozen = read_json(selection_path)
        if frozen["plan_hash"] != plan_hash:
            raise ValueError(f"확정된 계획과 실행 조건이 다릅니다. 새 실험에는 다른 --output-dir를 지정하세요. {selection_path}")
        selected_models = frozen["models"]
        print(f"검증으로 확정된 두 모델 설정을 복원했습니다. {selection_path}", flush=True)
    else:
        selected_models = tune_cnn_models(output_dir, plans)
        write_cnn_summary(output_dir, {"plan_hash": plan_hash, "models": selected_models}, selection_path.name)

    summaries = {}
    for model_id, selected in selected_models.items():
        model_dir = output_dir / model_id
        summary = selected["validation_summary"]
        summary_path = model_dir / "summary.json"
        previous_count = len(read_json(summary_path)["official_tests"]) if summary_path.is_file() else 0
        for final_config in cnn_final_configs(selected["config"]):
            result = run_cnn_trial(model_dir, final_config)
            report_path = cnn_report_path(model_dir, result.trial)
            if isinstance(result, FailedTrial):
                report_path = report_path.with_name("failure.json")
                entry = {**result.failure, "status": "failed", "accuracy": None, "loss": None, "num_test_samples": None}
            else:
                official_test = read_json(report_path)["final_test"]
                if (official_test["split"] != "test" or not 0 <= official_test["accuracy"] <= 1
                        or not math.isfinite(official_test["loss"])):
                    raise ValueError(f"공식 테스트 결과를 확인하세요. {report_path}")
                entry = {key: official_test[key] for key in ("accuracy", "loss", "num_test_samples")}
            summary["official_tests"].append({**entry, "seed": final_config["experiment"]["seed"],
                                              "report_path": report_path.relative_to(model_dir).as_posix()})
            if previous_count < len(summary["official_tests"]) < len(FINAL_SEEDS):
                write_cnn_summary(model_dir, summary)
        scores = [row["accuracy"] for row in summary["official_tests"] if row["accuracy"] is not None]
        summary["mean_test_accuracy"] = sum(scores) / len(FINAL_SEEDS) if len(scores) == len(FINAL_SEEDS) else None
        write_cnn_summary(model_dir, summary)
        summaries[model_id] = summary
    return summaries


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description="E02~E37 연합학습 독립 실험을 한 번에 실행합니다.")
    parser.add_argument("--cnn-only", action="store_true", help="SmallCNN(E03)과 4층 CNN(E04)을 독립적으로 튜닝합니다.")
    parser.add_argument("--device", choices=("cuda", "cpu", "auto"), help="CNN 전용 학습·평가 장치. NVIDIA GPU는 cuda를 지정하세요.")
    parser.add_argument("--dry-run", action="store_true", help="학습 없이 실행 순서와 결과 경로를 표시합니다.")
    parser.add_argument("--output-dir", type=Path, help="결과 저장 경로. 기본값은 모드에 따라 full_suite 또는 cnn_only입니다.")
    args = parser.parse_args(argv)
    if args.device and not args.cnn_only:
        parser.error("--device는 --cnn-only와 함께 사용하세요.")
    default_dir = CNN_OUTPUT_DIR if args.cnn_only else DEFAULT_OUTPUT_DIR
    output_dir = (args.output_dir or default_dir).resolve()
    if args.dry_run and not args.cnn_only:
        print("E02~E05 모델 → E06~E13 학습률 → E14~E17 드롭아웃 → E18~E22 증강")
        print("→ E23~E26 학습 일정 → E27 Adam → E28 드롭아웃 재검증 → E29~E37 최종 평가")
        print(f"총 36개 실험 | 결과 {output_dir}")
        return

    try:
        if args.cnn_only:
            if output_dir.is_relative_to(DEFAULT_OUTPUT_DIR.resolve()):
                raise ValueError("CNN 전용 결과 경로는 기존 full_suite 폴더 밖에 지정하세요.")
            plans = cnn_model_plans(args.device)
            if args.dry_run:
                print_cnn_plan(output_dir, plans)
            else:
                from filelock import FileLock, Timeout

                if args.device == "cuda":
                    import torch

                    if not torch.cuda.is_available():
                        raise ValueError("--device cuda를 실행하려면 CUDA를 사용할 수 있는 PyTorch 환경이 필요합니다.")
                    print(f"NVIDIA GPU | {torch.cuda.get_device_name(0)}", flush=True)
                workbook = load_cnn_workbook()
                output_dir.mkdir(parents=True, exist_ok=True)
                try:
                    with FileLock(output_dir / ".cnn-only.lock", timeout=0):
                        summaries = run_cnn_suite(output_dir, plans)
                        report_path = save_cnn_workbook(workbook, summaries, STAGE_LABELS, output_dir)
                        print(f"두 모델의 최종 비교표를 저장했습니다. {report_path}", flush=True)
                except Timeout as error:
                    raise ValueError(f"같은 폴더에서 CNN 실험이 이미 진행 중입니다. {output_dir}") from error
        else:
            check_dataset_files()
            run_suite(output_dir)
    except (ImportError, OSError, KeyError, TypeError, ValueError) as error:
        parser.exit(2, f"오류: {error}\n")
    except subprocess.CalledProcessError as error:
        parser.exit(error.returncode, "실험이 중단됐습니다. 위 오류와 결과 폴더를 확인하세요.\n")
    except KeyboardInterrupt:
        parser.exit(130, "실험을 중단했습니다. 같은 명령을 실행하면 저장된 라운드부터 이어서 진행합니다.\n")


if __name__ == "__main__":
    main()
