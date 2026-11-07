"""Sequential in-process role-1 runner, without Flask or network emulation."""

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import csv
import json
import math
import platform
from pathlib import Path
import random
import re
import time
from importlib.metadata import version
from uuid import uuid4

import torch
from torch.utils.data import DataLoader, Dataset

from .aggregation import fedavg, validate_updates
from .artifacts import load_checkpoint, save_checkpoint, save_update, write_json
from .attack import apply_collusion
from .config import ExperimentConfig, project_path
from .contracts import RoundContext
from .data import assigned_subset, load_cifar10, make_iid_partition, make_loader, validate_partition
from .model import build_model
from .tensors import TensorState, clone_state, state_hash, state_norm, validate_state
from .training import evaluate, resolve_device, seed_everything, train_local


def participation_schedule(config: ExperimentConfig) -> dict[str, list[int]]:
    schedule = {}
    count = max(1, math.ceil(config.num_clients * config.participation_rate))
    for round_id in range(1, config.num_rounds + 1):
        explicit = config.participation_schedule.get(str(round_id))
        selected = explicit if explicit is not None else random.Random(config.seed + round_id).sample(range(config.num_clients), count)
        schedule[str(round_id)] = sorted(selected)
    return schedule


def execution_plan(config: ExperimentConfig, mode: str) -> dict:
    if mode not in ("normal", "attack", "compare"):
        raise ValueError("mode는 normal/attack/compare입니다.")
    if mode in ("attack", "compare") and not config.attack.enabled:
        raise ValueError("attack/compare 모드에는 attack.enabled=true가 필요합니다.")
    return {"config": config.to_dict(), "mode": mode,
            "runs": ["normal", "attack"] if mode == "compare" else [mode],
            "participation_schedule": participation_schedule(config),
            "execution": "sequential virtual clients; no Flask/processes/timeouts",
            "defense_requires_role2_callback": config.defense.enabled}


class EventLog:
    def __init__(self, path: Path, experiment_id: str):
        self.path = path
        self.experiment_id = experiment_id

    def write(self, event: str, **payload):
        record = {"timestamp_utc": datetime.now(timezone.utc).isoformat(),
                  "experiment_id": self.experiment_id, "event": event, **payload}
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")


@dataclass(frozen=True)
class PreparedExperiment:
    """Data/model shared by normal and attack scenarios of one experiment."""

    device: torch.device
    datasets: dict[int, Dataset]
    evaluation_loader: DataLoader
    initial: TensorState
    initial_version: int


def prepare_experiment(config: ExperimentConfig, plan: dict, run_dir: Path) -> PreparedExperiment:
    """Prepare datasets and starting model, then save reproducibility artifacts."""
    device = resolve_device(config.runtime)
    # No implicit download: use the download-data command explicitly.
    source = load_cifar10(config.data, train=True)
    if config.data.partition_file:
        partition = json.loads(project_path(config.data.partition_file).read_text(encoding="utf-8-sig"))
        validate_partition(partition, list(source.targets), num_clients=config.num_clients)
    else:
        partition = make_iid_partition(list(source.targets), num_clients=config.num_clients,
                                       validation_size=config.data.validation_size, seed=config.seed)
    if config.evaluation.split == "validation":
        evaluation_data = assigned_subset(source, partition["validation_indices"], config.evaluation.max_samples)
    else:
        test = load_cifar10(config.data, train=False)
        evaluation_data = assigned_subset(test, list(range(len(test))), config.evaluation.max_samples)
    evaluation_loader = make_loader(evaluation_data, batch_size=config.evaluation.batch_size,
                                    shuffle=False, seed=config.seed, num_workers=config.runtime.num_workers)
    datasets = {i: assigned_subset(source, partition["client_indices"][str(i)], config.data.max_train_samples_per_client)
                for i in range(config.num_clients)}
    seed_everything(config.seed, deterministic=config.runtime.deterministic)
    initial_version = 0
    if config.initial_checkpoint:
        checkpoint = load_checkpoint(project_path(config.initial_checkpoint))
        if checkpoint["model_config"] != asdict(config.model):
            # JSON channels are lists, dataclasses may hold tuples.
            if json.dumps(checkpoint["model_config"], sort_keys=True) != json.dumps(asdict(config.model), sort_keys=True):
                raise ValueError("초기 체크포인트의 모델 설정이 다릅니다.")
        preprocessing = checkpoint["preprocessing"]
        if list(config.data.normalization_mean) != list(preprocessing["normalization_mean"]) or list(config.data.normalization_std) != list(preprocessing["normalization_std"]):
            raise ValueError("초기 체크포인트의 전처리 설정이 다릅니다.")
        initial = clone_state(checkpoint["model_state"])
        initial_version = checkpoint["model_version"]
    else:
        initial = clone_state(build_model(config.model).state_dict())
    # Retain exist_ok=False as a second guard against concurrent executions.
    run_dir.mkdir(parents=True, exist_ok=False)
    write_json(run_dir / "config.json", {**plan, "versions": {"python": platform.python_version(),
               **{package: version(package) for package in ("torch", "torchvision", "numpy", "Pillow")}},
               "device_used": str(device), "accuracy_unit": "fraction_0_to_1"})
    write_json(run_dir / "partition.json", partition)
    write_json(run_dir / "used_indices.json", {
        "train": {str(i): list(dataset.indices) for i, dataset in datasets.items()},
        "evaluation": {"split": config.evaluation.split, "indices": list(evaluation_data.indices)}})
    save_checkpoint(run_dir / "initial.pt", state=initial, model_config=config.model,
                    data_config=config.data, experiment_id=config.experiment_id, model_version=initial_version)
    return PreparedExperiment(device, datasets, evaluation_loader, initial, initial_version)


def run_round(config: ExperimentConfig, prepared: PreparedExperiment, *,
              current: TensorState, context: RoundContext, scenario: str,
              scenario_dir: Path, events: EventLog, train_fn, evaluate_fn,
              aggregation_fn=None) -> tuple[TensorState, dict]:
    """Run local learning, attack, aggregation and evaluation for one round."""
    round_id, current_version = context.round_id, context.model_version
    selected = context.selected_clients
    datasets = prepared.datasets
    evaluation_loader, device = prepared.evaluation_loader, prepared.device
    started = time.perf_counter()
    events.write("round_start", round_id=round_id, model_version=current_version,
                 selected=list(selected), base_model_hash=state_hash(current))
    round_dir = scenario_dir / f"round_{round_id:04d}"
    round_dir.mkdir()
    clean = []
    for client_id in selected:
        def on_epoch(row, client_id=client_id):
            events.write("local_training", round_id=round_id, client_id=client_id, **row)
        local = train_fn(current, datasets[client_id], client_id=client_id, context=context,
                         model_config=config.model, training_config=config.training,
                         runtime=config.runtime, seed=config.seed, on_epoch=on_epoch)
        clean.append(local.update)
        save_update(round_dir / "clean_updates" / f"client_{client_id}.pt", local.update)
        write_json(round_dir / "local_history" / f"client_{client_id}.json", {"epochs": local.history})
    validate_updates(current, clean, context)
    for update in clean:
        if update.num_samples != len(datasets[update.client_id]):
            raise ValueError("업데이트 표본 수가 배정 데이터 수와 다릅니다.")
    attack_applied, attack_reason = False, "normal_scenario"
    submitted = clean
    if scenario == "attack":
        outcome = apply_collusion(clean, config.attack, selected)
        submitted, attack_applied, attack_reason = outcome.updates, outcome.applied, outcome.reason
    events.write("attack_decision", round_id=round_id, applied=attack_applied, reason=attack_reason,
                 participating_malicious_clients=sorted(set(selected) & set(config.attack.malicious_clients)))
    for update in submitted:
        save_update(round_dir / "submitted_updates" / f"client_{update.client_id}.pt", update)
        events.write("update_submitted", **update.metadata(), update_norm=state_norm(update.delta))
    validate_updates(current, submitted, context)
    if config.defense.enabled:
        updated = aggregation_fn(clone_state(current), submitted, context, config.defense)
        validate_state(updated, current)
        current = clone_state(updated)
    else:
        current = fedavg(current, submitted, context)
    current_version += 1
    metrics = evaluate_fn(build_model(config.model, current).to(device), evaluation_loader, device)
    save_checkpoint(round_dir / "global.pt", state=current, model_config=config.model,
                    data_config=config.data, experiment_id=config.experiment_id, model_version=current_version)
    result = {"experiment_id": config.experiment_id, "round_id": round_id,
              "model_version": current_version, "status": "completed", "metrics": metrics,
              "clients": {"selected": list(selected), "received": [u.client_id for u in submitted], "excluded": []},
              "evaluation_split": config.evaluation.split, "execution_time": time.perf_counter() - started}
    write_json(round_dir / "result.json", result)
    events.write("round_complete", **result)
    row = {"round_id": round_id, "model_version": current_version, **metrics,
           "evaluation_split": config.evaluation.split, "attack_applied": attack_applied,
           "execution_time": result["execution_time"]}
    return current, row


def simulate(config: ExperimentConfig, *, mode: str, run_name: str | None = None,
             aggregation_fn=None, train_fn=None, evaluate_fn=None) -> Path:
    """aggregation_fn(W, updates, context, defense) is the role-2 extension.

    It never receives attack settings. train_fn/evaluate_fn support integration
    tests with stubs; the command-line always uses actual training/evaluation.
    """
    plan = execution_plan(config, mode)
    if config.defense.enabled and aggregation_fn is None:
        raise ValueError("방어는 2번 담당자의 aggregation_fn을 연결한 뒤 실행하세요.")
    if run_name is not None and not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", run_name):
        raise ValueError("run-name은 영문/숫자로 시작하는 최대 64자의 영문/숫자/_/- 이름입니다.")
    name = run_name or f"run_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}_{uuid4().hex[:6]}"
    run_dir = project_path(config.runtime.output_root) / name
    if run_dir.exists():
        raise FileExistsError(f"이미 존재하는 결과 폴더입니다. 새 run-name을 지정하세요: {run_dir}")
    train_fn = train_fn or train_local
    evaluate_fn = evaluate_fn or evaluate
    prepared = prepare_experiment(config, plan, run_dir)
    summaries = {}
    for scenario in plan["runs"]:
        scenario_dir = run_dir / scenario
        scenario_dir.mkdir()
        events = EventLog(scenario_dir / "events.jsonl", config.experiment_id)
        current = clone_state(prepared.initial)
        current_version = prepared.initial_version
        rows = []
        try:
            for round_id in range(1, config.num_rounds + 1):
                context = RoundContext(config.experiment_id, round_id, current_version,
                                       tuple(plan["participation_schedule"][str(round_id)]))
                current, row = run_round(
                    config, prepared, current=current, context=context, scenario=scenario,
                    scenario_dir=scenario_dir, events=events, train_fn=train_fn,
                    evaluate_fn=evaluate_fn, aggregation_fn=aggregation_fn,
                )
                current_version = row["model_version"]
                rows.append(row)
                # Keep completed-round metrics even if a later round fails.
                with (scenario_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as stream:
                    writer = csv.DictWriter(stream, fieldnames=list(row))
                    writer.writeheader()
                    writer.writerows(rows)
                print(f"{scenario} round {round_id}/{config.num_rounds} | {config.evaluation.split} "
                      f"accuracy={row['accuracy']:.2%}, loss={row['loss']:.4f}, attack={row['attack_applied']}", flush=True)
            summaries[scenario] = {"status": "completed", "final": rows[-1], "initial_model_hash": state_hash(prepared.initial)}
            write_json(scenario_dir / "summary.json", summaries[scenario])
        except Exception as error:
            events.write("failed", error_type=type(error).__name__, message=str(error))
            write_json(scenario_dir / "summary.json", {"status": "failed", "completed_rounds": len(rows), "error": str(error)})
            raise
    if mode == "compare":
        write_json(run_dir / "comparison.json", {
            "normal": summaries["normal"], "attack": summaries["attack"],
            "accuracy_drop": summaries["normal"]["final"]["accuracy"] - summaries["attack"]["final"]["accuracy"],
            "accuracy_drop_unit": "fraction_0_to_1", "shared_initial_model": True,
            "shared_partition_and_participation": True})
    print(f"저장 완료: {run_dir.resolve()}", flush=True)
    return run_dir
