"""독립 실행 CLI. python -m fedsec --config configs/fedsec.example.json"""

import argparse
import json
from pathlib import Path
import re
import torch

from fedsec.config import ExperimentSettings
from fedsec.data import load_datasets
from fedsec.experiment import run_comparison_experiment
from fedsec.model import MODEL_FORMAT, PREPROCESSING
from fedsec.tensors import state_hash


def _write_json(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)


def main():
    parser = argparse.ArgumentParser(description="Flask 없이 정상/공격 연합학습을 실제 실행합니다.")
    parser.add_argument("--config", type=Path, required=True, help="실험 설정 및 데이터 배정표 JSON")
    parser.add_argument("--experiment-id", help="JSON의 experiment_id 덮어쓰기")
    parser.add_argument("--output-dir", type=Path, default=Path("results/standalone"))
    args = parser.parse_args()
    try:
        payload = json.loads(args.config.read_text(encoding="utf-8-sig"))
        unknown = payload.keys() - {"experiment_id", "experiment", "datasets"}
        if unknown:
            raise ValueError(f"Unknown configuration keys: {sorted(unknown)}")
        experiment_id = args.experiment_id or payload["experiment_id"]
        if not isinstance(experiment_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,93}", experiment_id):
            raise ValueError("experiment_id는 영문·숫자·밑줄·하이픈 1~93자로 입력하세요.")
        config = ExperimentSettings.from_dict(payload["experiment"])
        directory = args.output_dir / experiment_id
        if directory.exists():
            raise FileExistsError(f"새 실험 ID를 사용하세요. 기존 결과: {directory}")
        clients, test = load_datasets(payload["datasets"], config.training)
        if len(config.malicious_clients) < 2 or not set(config.malicious_clients) <= clients.keys():
            raise ValueError("공격 클라이언트 2개 이상이 데이터 배정표에 있어야 합니다.")
        directory.mkdir(parents=True, exist_ok=False)
        _write_json(directory / "config.json", {**payload, "experiment_id": experiment_id})

        def record_round(record):
            phase = "attack" if record["attack_enabled"] else "normal"
            _write_json(directory / phase / "rounds" / f"round_{record['round']:03d}.json", record)
            print(f"[{phase}] {record['round']}/{config.num_rounds} "
                  f"accuracy={record['accuracy']:.4f}, loss={record['loss']:.4f}", flush=True)

        print(f"가상 클라이언트 {len(clients)}개를 순차 실행합니다. 실제 학습을 시작합니다.", flush=True)
        result = run_comparison_experiment(config, clients, test, experiment_id=experiment_id, on_round=record_round)
        for phase, run in (("normal", result.normal), ("attack", result.attacked)):
            _write_json(directory / phase / "results.json", run.report())
            # 기존 common.model.load_model_checkpoint()로도 읽을 수 있는 체크포인트.
            torch.save({"format_version": 1, "architecture": MODEL_FORMAT,
                        "model_config": config.model_config, "preprocessing": PREPROCESSING,
                        "model_version": config.num_rounds, "experiment_id": run.config["experiment_id"],
                        "seed": config.seed, "initial_model_hash": run.round_results[0]["initial_model_hash"],
                        "model_hash": state_hash(run.final_weights), "weights": run.final_weights},
                       directory / phase / "final_model.pt")
        _write_json(directory / "comparison.json", result.comparison)
        print(f"비교 완료: {directory / 'comparison.json'}", flush=True)
    except (OSError, ValueError, TypeError, KeyError, RuntimeError) as error:
        parser.exit(2, f"오류: {error}\n")
    except KeyboardInterrupt:
        parser.exit(130, "실행을 중단했습니다. 완료된 라운드 JSON은 결과 폴더에 남아 있습니다.\n")


if __name__ == "__main__":
    main()
