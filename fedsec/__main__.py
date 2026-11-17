"""독립 실행 CLI. python -m fedsec --config configs/fedsec.example.json"""

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import re
from tempfile import NamedTemporaryFile
from time import perf_counter

from filelock import FileLock, Timeout
import torch

from fedsec.checkpoint import load_checkpoint
from fedsec.comparison import compare_accuracy_results
from fedsec.config import ExperimentSettings
from fedsec.data import CifarRangeDataset, load_datasets
from fedsec.experiment import RoundState, run_comparison_experiment, run_federated_learning
from fedsec.model import MODEL_FORMAT, PREPROCESSING, create_model
from fedsec.tensors import state_hash
from fedsec.training import evaluate_global_model


def _write_json(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)


def _replace_file(path: Path, write):
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, delete=False) as file:
        pending = Path(file.name)
    try:
        write(pending)
        pending.replace(path)
    finally:
        pending.unlink(missing_ok=True)


def _replace_json(path: Path, payload: dict):
    content = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False)
    _replace_file(path, lambda pending: pending.write_text(content, encoding="utf-8"))


def _model_checkpoint(config: ExperimentSettings, record: dict, weights: dict) -> dict:
    return {"format_version": 1, "architecture": config.model_config.get("architecture", MODEL_FORMAT),
            "model_config": config.model_config, "preprocessing": PREPROCESSING,
            "model_version": record["round"], "experiment_id": record["experiment_id"],
            "seed": config.seed, "initial_model_hash": record["initial_model_hash"],
            "model_hash": state_hash(weights), "weights": weights}


def _read_config(path: Path, experiment_id_override: str | None) -> tuple[dict, ExperimentSettings]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError("실험 설정은 JSON 객체여야 합니다.")
    unknown = payload.keys() - {"experiment_id", "experiment", "datasets", "mode", "evaluation_split"}
    if unknown:
        raise ValueError(f"Unknown configuration keys: {sorted(unknown)}")
    mode = payload.get("mode", "compare")
    if mode not in ("normal", "attack", "compare"):
        raise ValueError("mode must be normal, attack or compare")
    evaluation_split = payload.get("evaluation_split", "test")
    if evaluation_split not in ("validation", "test"):
        raise ValueError("evaluation_split must be validation or test")
    experiment_id = experiment_id_override or payload["experiment_id"]
    if not isinstance(experiment_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,93}", experiment_id):
        raise ValueError("experiment_id는 영문·숫자·밑줄·하이픈 1~93자로 입력하세요.")
    experiment = payload["experiment"]
    if not isinstance(experiment, dict) or not isinstance(experiment.get("training", {}), dict):
        raise ValueError("experiment와 training은 JSON 객체여야 합니다.")
    settings = ExperimentSettings.from_dict(experiment)
    datasets = payload["datasets"]
    if isinstance(datasets, str) and datasets:
        manifest_path = path.parent / datasets
        datasets = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    if not isinstance(datasets, dict):
        raise ValueError("datasets에는 데이터 배정표 객체 또는 JSON 파일 경로를 입력하세요.")
    if "final_test_ranges" in datasets and (evaluation_split != "validation" or not datasets["final_test_ranges"]):
        raise ValueError("최종 테스트 범위를 지정하려면 라운드 평가에 검증셋을 사용해야 합니다.")
    return {**payload, "experiment_id": experiment_id, "mode": mode,
            "evaluation_split": evaluation_split, "datasets": datasets}, settings


def _run_config(payload: dict, config: ExperimentSettings, directory: Path, *, resume: bool = False):
    directory.mkdir(parents=True, exist_ok=resume)
    try:
        with FileLock(directory / ".run.lock", timeout=0):
            saved_config = directory / "config.json"
            if saved_config.exists():
                if not resume or _read_config(saved_config, None)[0] != payload:
                    raise ValueError(f"저장된 설정과 실행 설정이 일치하지 않습니다. {saved_config}")
            else:
                _replace_json(saved_config, payload)
            _train_and_save(payload, config, directory, resume=resume)
    except Timeout as error:
        raise ValueError(f"같은 결과 폴더에서 실험이 이미 진행 중입니다. {directory}") from error


def _train_and_save(payload: dict, config: ExperimentSettings, directory: Path, *, resume: bool):
    experiment_id = payload["experiment_id"]
    mode = payload["mode"]
    split = payload["evaluation_split"]
    manifest = payload["datasets"]
    phase_dir = directory / mode
    progress_path = phase_dir / "round_checkpoint.pt"
    if resume and (phase_dir / "results.json").is_file():
        print(f"[{experiment_id}] 완료된 결과 사용", flush=True)
        return

    progress = None
    elapsed_before = 0.0
    config_hash = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
    if resume and progress_path.is_file():
        saved = load_checkpoint(progress_path)
        records = saved["round_results"]
        if (saved["config_hash"] != config_hash or saved["model_version"] != len(records)
                or saved["experiment_id"] != experiment_id):
            raise ValueError(f"체크포인트가 현재 실험 설정과 일치하지 않습니다. {progress_path}")
        progress = RoundState(saved["weights"], records)
        elapsed_before = saved["elapsed_seconds"]
    elif resume and any((phase_dir / "rounds").glob("round_*.json")):
        raise ValueError(f"{experiment_id}의 기존 결과에는 재개용 가중치가 없습니다. "
                         f"{directory} 폴더를 다른 곳에 보관한 뒤 다시 실행하세요.")

    clients, evaluation_data = load_datasets(manifest, config.training)
    final_test = None
    if "final_test_ranges" in manifest:
        final_test = CifarRangeDataset({"source_dir": manifest["source_dir"], "ranges": manifest["final_test_ranges"]})
    if mode != "normal" and (
        len(config.malicious_clients) < 2 or not set(config.malicious_clients) <= clients.keys()
    ):
        raise ValueError("공격 클라이언트 2개 이상이 데이터 배정표에 있어야 합니다.")
    def record_round(record):
        phase = "attack" if record["attack_enabled"] else "normal"
        record["evaluation"]["split"] = split
        path = directory / phase / "rounds" / f"round_{record['round']:03d}.json"
        _replace_json(path, record)
        print(f"[{experiment_id}/{phase}] {record['round']}/{config.num_rounds} "
              f"{split} accuracy={record['accuracy']:.4f}, loss={record['loss']:.4f}", flush=True)

    def save_round(state: RoundState):
        checkpoint = _model_checkpoint(config, state.records[-1], state.weights)
        checkpoint.update(config_hash=config_hash, round_results=state.records,
                          elapsed_seconds=elapsed_before + perf_counter() - started)
        _replace_file(progress_path, lambda pending: torch.save(checkpoint, pending))

    if progress is None:
        print(f"[{experiment_id}] 가상 클라이언트 {len(clients)}개 학습을 시작합니다.", flush=True)
    else:
        print(f"[{experiment_id}] {len(progress.records)}/{config.num_rounds} 라운드 완료 상태를 불러왔습니다.", flush=True)
    options = {"experiment_id": experiment_id, "on_round": record_round}
    started = perf_counter()
    if mode == "compare":
        result = run_comparison_experiment(config, clients, evaluation_data, **options)
        runs = (("normal", result.normal), ("attack", result.attacked))
    else:
        run = run_federated_learning(config, clients, evaluation_data, attack=(mode == "attack"),
                                     resume=progress, save_round=save_round, **options)
        runs = ((mode, run),)
    elapsed = elapsed_before + perf_counter() - started

    reports = {}
    for phase, run in runs:
        if progress is not None:
            for record in progress.records:
                log_path = phase_dir / "rounds" / f"round_{record['round']:03d}.json"
                if not log_path.is_file():
                    record_round(record)
        checkpoint_path = directory / phase / "final_model.pt"
        checkpoint = _model_checkpoint(config, run.round_results[-1], run.final_weights)
        _replace_file(checkpoint_path, lambda pending: torch.save(checkpoint, pending))
        report = run.report()
        report["evaluation_split"] = split
        for record in report["round_results"]:
            record["evaluation"]["split"] = split
        if final_test is not None:
            model = create_model(config.seed, config.model_config)
            model.load_state_dict(run.final_weights)
            evaluation = evaluate_global_model(model, final_test, replace(config.training, max_test_samples=None),
                                               seed=config.seed)
            report["final_test"] = {"split": "test", "round": config.num_rounds, "model_version": config.num_rounds,
                                    "accuracy": evaluation.accuracy, "loss": evaluation.loss,
                                    "num_test_samples": evaluation.num_samples, "correct_predictions": evaluation.correct_predictions,
                                    "dataset_fingerprint": final_test.fingerprint, "checkpoint": str(checkpoint_path),
                                    "metrics": {"per_class": evaluation.per_class}}
            print(f"[{phase}] 최종 테스트 정확도 {evaluation.accuracy:.4f} ({evaluation.num_samples}장)", flush=True)
        if mode != "compare":
            load_checkpoint(checkpoint_path)
            report.update(elapsed_seconds=elapsed, checkpoint_verified=True)
            print(f"체크포인트 복원 확인: {checkpoint_path}", flush=True)
            print(f"학습·평가 시간: {elapsed:.2f}초", flush=True)
        _replace_json(directory / phase / "results.json", report)
        reports[phase] = report
    if mode == "compare":
        if final_test is not None:
            comparison = compare_accuracy_results(reports["normal"], reports["attack"])
        else:
            comparison = result.comparison
        _write_json(directory / "comparison.json", comparison)
        print(f"비교 완료: {directory / 'comparison.json'}", flush=True)
    else:
        progress_path.unlink(missing_ok=True)
        print(f"[{mode}] 학습 완료: {directory / mode / 'results.json'}", flush=True)


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description="실험 설정을 읽어 Flask 없이 연합학습을 순서대로 실행합니다.")
    parser.add_argument("--config", action="append", type=Path, required=True,
                        help="실험 설정 JSON. 여러 실험은 --config를 반복해 지정합니다.")
    parser.add_argument("--experiment-id", help="설정 파일 하나를 실행할 때 실험 ID를 덮어씁니다.")
    parser.add_argument("--output-dir", type=Path, default=Path("results/standalone"))
    parser.add_argument("--dry-run", action="store_true", help="학습 없이 설정을 읽고 실행 순서를 확인합니다.")
    parser.add_argument("--resume", action="store_true", help="같은 실험의 저장된 라운드부터 이어서 실행합니다.")
    args = parser.parse_args(argv)
    if args.experiment_id is not None and len(args.config) != 1:
        parser.error("--experiment-id는 --config가 하나일 때만 사용할 수 있습니다.")
    try:
        runs = []
        planned_directories = set()
        for path in args.config:
            try:
                payload, config = _read_config(path, args.experiment_id)
            except (OSError, ValueError, TypeError, KeyError) as error:
                raise ValueError(f"{path}: {error}") from error
            directory = (args.output_dir / payload["experiment_id"]).resolve()
            if directory in planned_directories:
                raise ValueError(f"같은 결과 폴더를 사용하는 실험이 있습니다: {directory}")
            if args.resume and payload["mode"] == "compare":
                raise ValueError("--resume은 normal 또는 attack 실험에 사용할 수 있습니다.")
            if directory.exists() and not args.resume:
                raise FileExistsError(f"새 실험 ID를 사용하세요. 기존 결과: {directory}")
            planned_directories.add(directory)
            runs.append((payload, config, directory))

        for payload, config, directory in runs:
            if args.dry_run:
                architecture = config.model_config.get("architecture", MODEL_FORMAT)
                print(f"{payload['experiment_id']} | 모델 {architecture} | {config.num_rounds}라운드 "
                      f"× 로컬 {config.training.local_epochs} epoch | 학습률 {config.training.learning_rate:g} "
                      f"| 드롭아웃 {config.model_config.get('dropout', 0):g} "
                      f"| {config.training.optimizer.upper()} | 증강 {config.training.augmentation} "
                      f"| 시드 {config.seed} | 모드 {payload['mode']} | 평가 {payload['evaluation_split']} "
                      f"{'| 최종 테스트 포함 ' if 'final_test_ranges' in payload['datasets'] else ''}"
                      f"| 결과 {directory}")
            else:
                _run_config(payload, config, directory, resume=args.resume)
    except (OSError, ValueError, TypeError, KeyError, RuntimeError) as error:
        parser.exit(2, f"오류: {error}\n")
    except KeyboardInterrupt:
        parser.exit(130, "실행을 중단했습니다. 저장된 라운드 결과는 결과 폴더에 있습니다.\n")


if __name__ == "__main__":
    main()
