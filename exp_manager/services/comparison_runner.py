"""정상 학습 → 공격 학습 → 비교를 기존 매니저 REST API로 실행한다.

import만으로는 학습하지 않는다. 서버/클라이언트 시작 및 데이터 분배는 먼저
기존 매니저에서 완료해야 하며, 이 함수 실행 중에는 웹에서 설정을 바꾸지 않는다.
"""

import argparse
from copy import deepcopy
import json
from pathlib import Path
import re
import time
from urllib.error import URLError

from common.comparison import compare_accuracy_results
from common.model import finite_number, positive_int, model_config
from common.training import TrainingSettings
from exp_manager.services.rest_client import RestClient
from exp_manager.utils.project_root_path import PROJECT_ROOT_PATH


def run_federated_learning(config: dict, *, manager_url: str = "http://127.0.0.1:8000",
                           poll_interval: float = 1.0, timeout: float | None = None) -> dict:
    """한 실험의 모든 라운드가 완료될 때까지 기다리고 비교 가능한 결과를 반환한다.

    각 라운드의 로컬 학습·공격·FedAvg·평가는 기존 클라이언트/서버가 수행한다.
    중단/실패 시 예외를 발생시켜 다음 실험을 실행하지 않는다.
    """
    finite_number(poll_interval, "poll_interval", positive=True)
    rounds = positive_int(config["num_rounds"], "num_rounds")
    custom = config.get("round_configs", [])
    budget = sum(float(custom[i].get("round_timeout", config["round_timeout"]))
                 if i < len(custom) else float(config["round_timeout"])
                 for i in range(rounds)) + 120.0 * rounds + 60.0
    timeout = finite_number(timeout if timeout is not None else budget, "timeout", positive=True)
    rest = RestClient()
    base = manager_url.rstrip("/")
    state = rest.get_json(f"{base}/api/experiment/status")
    if state["status"] in ("running", "stopping"):
        raise RuntimeError("이미 실행 중인 실험이 있습니다. 완료 후 비교를 시작하세요.")
    rest.post_json(f"{base}/api/experiment/config", deepcopy(config))
    state = rest.post_json(f"{base}/api/experiment/start", {}, timeout=30.0)
    deadline = time.monotonic() + timeout
    try:
        while True:
            if state.get("config", {}).get("experiment_id") != config["experiment_id"]:
                raise RuntimeError("매니저의 실험 ID가 변경되었습니다. 비교 중 다른 실험을 시작하지 마세요.")
            status = state["status"]
            if status == "completed":
                records = state.get("round_results", [])
                if sorted(item["round"] for item in records) != list(range(1, rounds + 1)):
                    raise RuntimeError("완료된 라운드가 누락되었거나 중복되었습니다.")
                return {"config": deepcopy(state["config"]), "round_results": deepcopy(records),
                        "result_path": state.get("result_path")}
            if status != "running":
                raise RuntimeError(f"실험이 완료되지 않았습니다 ({status}): {state.get('message', '')}")
            if time.monotonic() >= deadline:
                raise TimeoutError("실험 대기 시간이 초과되었습니다.")
            time.sleep(min(poll_interval, max(0.0, deadline - time.monotonic())))
            state = rest.get_json(f"{base}/api/experiment/status")
    except (KeyboardInterrupt, TimeoutError):
        # 다른 사용자가 시작한 실험에는 중단 요청을 보내지 않는다.
        try:
            current = rest.get_json(f"{base}/api/experiment/status")
            if (current.get("config", {}).get("experiment_id") == config["experiment_id"]
                    and current["status"] == "running"):
                rest.post_json(f"{base}/api/experiment/stop", {})
        except (OSError, ValueError):
            pass
        raise


def run_comparison_experiment(config: dict, *, manager_url: str = "http://127.0.0.1:8000",
                              output_dir: Path | str | None = None,
                              poll_interval: float = 1.0, timeout: float | None = None) -> dict:
    """1번 담당의 통합 진입점: 정상 FL → 공격 FL → 정확도 비교 및 JSON 저장.

    config는 매니저 설정의 덮어쓰기 값이다. experiment_id는 비교 이름이며
    실제 두 실험 ID에는 _normal, _attack을 붙인다. 동일한 설정 스냅샷을 사용하고
    두 실험 모두 방어를 끄므로 FedAvg에 대한 내부자 공격 효과를 비교한다.
    데이터는 재분배하지 않는다. 참여/제출/데이터/초기 모델이 달라지면 비교를 거부한다.
    """
    prefix = config.get("experiment_id", "")
    if not isinstance(prefix, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,93}", prefix):
        raise ValueError("비교 experiment_id는 영문·숫자·밑줄·하이픈 1~93자로 입력하세요.")
    finite_number(poll_interval, "poll_interval", positive=True)
    if timeout is not None:
        finite_number(timeout, "timeout", positive=True)
    output = Path(output_dir) if output_dir is not None else PROJECT_ROOT_PATH / "results"
    report_path = output / f"{prefix}_comparison.json"
    if report_path.exists():
        raise FileExistsError(f"비교 결과가 이미 있습니다. 새 ID를 사용하세요: {report_path}")

    rest = RestClient()
    base = manager_url.rstrip("/")
    shared = deepcopy({**rest.get_json(f"{base}/api/experiment/config"), **config})
    TrainingSettings.from_dict(shared)
    model_config(shared.get("model_config"))
    positive_int(shared["num_rounds"], "num_rounds")
    finite_number(shared["round_timeout"], "round_timeout", positive=True)
    if shared.get("experiment_type") != "federated" or any(
            item.get("round_type", "federated") != "federated" for item in shared.get("round_configs", [])):
        raise ValueError("이 함수는 일반 연합학습과 내부자 공격 비교용입니다. DLG는 지원하지 않습니다.")

    def condition(attack: bool) -> dict:
        result = deepcopy(shared)
        forced = {"attack_enabled": attack, "defense_enabled": False,
                  "defense": "none", "round_type": "federated"}
        result.update(forced, experiment_id=f"{prefix}_{'attack' if attack else 'normal'}")
        result["round_configs"] = [{**item, **forced} for item in result.get("round_configs", [])]
        return result

    options = {"manager_url": base, "poll_interval": poll_interval, "timeout": timeout}
    normal = run_federated_learning(condition(False), **options)
    attacked = run_federated_learning(condition(True), **options)
    comparison = compare_accuracy_results(normal, attacked)
    comparison["normal_result_path"] = normal.get("result_path")
    comparison["attack_result_path"] = attacked.get("result_path")
    comparison["comparison_path"] = str(report_path)
    output.mkdir(parents=True, exist_ok=True)
    with report_path.open("x", encoding="utf-8") as handle:
        json.dump(comparison, handle, ensure_ascii=False, indent=2, allow_nan=False)
    return comparison


def run_full_security_comparison(
    config: dict,
    seeds: list[int],
    *,
    manager_url: str = "http://127.0.0.1:8000",
    output_dir: Path | str | None = None,
    poll_interval: float = 1.0,
    timeout: float | None = None,
) -> dict:
    """정상·공격·두 방어 방식의 6개 조건을 시드별로 실행해 피해/개선 지표를 저장한다."""
    prefix = config.get("experiment_id", "")
    if not isinstance(prefix, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,55}", prefix):
        raise ValueError("비교 experiment_id는 영문·숫자·밑줄·하이픈 1~55자로 입력하세요.")
    if not isinstance(seeds, list) or not seeds:
        raise ValueError("비교할 시드를 하나 이상 지정하세요.")
    normalized_seeds = [positive_int(seed, "seed", 0) for seed in seeds]
    if len(set(normalized_seeds)) != len(normalized_seeds):
        raise ValueError("비교 시드는 중복될 수 없습니다.")
    finite_number(poll_interval, "poll_interval", positive=True)
    if timeout is not None:
        finite_number(timeout, "timeout", positive=True)

    rest = RestClient()
    base = manager_url.rstrip("/")
    shared = deepcopy({**rest.get_json(f"{base}/api/experiment/config"), **config})
    TrainingSettings.from_dict(shared)
    model_config(shared.get("model_config"))
    positive_int(shared["num_rounds"], "num_rounds")
    finite_number(shared["round_timeout"], "round_timeout", positive=True)
    if shared.get("experiment_type") != "federated" or any(
        item.get("round_type", "federated") != "federated"
        for item in shared.get("round_configs", [])
    ):
        raise ValueError("정상·공격·방어 비교는 연합학습 라운드만 지원합니다.")

    output = Path(output_dir) if output_dir is not None else PROJECT_ROOT_PATH / "results"
    report_path = output / f"{prefix}_security_comparison.json"
    if report_path.exists():
        raise FileExistsError(f"비교 결과가 이미 있습니다. 새 ID를 사용하세요: {report_path}")

    scenario_settings = {
        "normal": (False, False, "none"),
        "attack": (True, False, "none"),
        "normal_defense_norm_clipping": (False, True, "norm_clipping"),
        "attack_defense_norm_clipping": (True, True, "norm_clipping"),
        "normal_defense_coordinate_median": (False, True, "coordinate_median"),
        "attack_defense_coordinate_median": (True, True, "coordinate_median"),
    }
    seed_rows = []
    run_paths = {}
    options = {"manager_url": base, "poll_interval": poll_interval, "timeout": timeout}
    for seed in normalized_seeds:
        runs = {}
        for scenario, (attack_enabled, defense_enabled, defense) in scenario_settings.items():
            case_config = deepcopy(shared)
            experiment_id = f"{prefix}-s{seed}-{scenario}"
            forced = {
                "attack_enabled": attack_enabled,
                "defense_enabled": defense_enabled,
                "defense": defense,
                "round_type": "federated",
            }
            case_config.update(
                forced,
                seed=seed,
                experiment_id=experiment_id,
                round_configs=[
                    {**item, **forced} for item in case_config.get("round_configs", [])
                ],
            )
            runs[scenario] = run_federated_learning(case_config, **options)
            run_paths[experiment_id] = runs[scenario].get("result_path")

        final = {
            scenario: result["round_results"][-1]
            for scenario, result in runs.items()
        }
        reference = final["normal"]
        for scenario, result in final.items():
            for key in ("initial_model_hash", "training_config", "training_datasets"):
                if not reference.get(key) or result.get(key) != reference.get(key):
                    raise ValueError(
                        f"시드 {seed}의 {scenario}: {key}가 정상 조건과 다릅니다."
                    )
            if any(not item.get("fingerprint") for item in result["training_datasets"].values()):
                raise ValueError("학습 데이터 fingerprint가 없어 공정 비교할 수 없습니다.")
            if (
                sorted(result.get("selected_clients", []))
                != sorted(reference.get("selected_clients", []))
                or sorted(result.get("accepted_clients", []))
                != sorted(reference.get("accepted_clients", []))
            ):
                raise ValueError(f"시드 {seed}의 {scenario}: 선택/채택 클라이언트가 다릅니다.")
            evaluation = result.get("evaluation", {})
            reference_evaluation = reference.get("evaluation", {})
            if (
                not evaluation.get("dataset_fingerprint")
                or evaluation.get("dataset_fingerprint")
                != reference_evaluation.get("dataset_fingerprint")
                or evaluation.get("num_test_samples", 0) <= 0
                or evaluation.get("num_test_samples")
                != reference_evaluation.get("num_test_samples")
            ):
                raise ValueError(f"시드 {seed}의 {scenario}: 테스트 데이터가 다릅니다.")

        accuracy = {
            name: finite_number(result["accuracy"], f"{name}.accuracy")
            for name, result in final.items()
        }
        defense_results = {}
        for defense in ("norm_clipping", "coordinate_median"):
            defended_attack = accuracy[f"attack_defense_{defense}"]
            defended_normal = accuracy[f"normal_defense_{defense}"]
            defense_results[defense] = {
                "attack_impact_percentage_points": (accuracy["normal"] - accuracy["attack"]) * 100,
                "defense_improvement_percentage_points": (defended_attack - accuracy["attack"]) * 100,
                "normal_defense_loss_percentage_points": (accuracy["normal"] - defended_normal) * 100,
                "accuracies": {
                    "normal": accuracy["normal"],
                    "attack": accuracy["attack"],
                    "normal_defense": defended_normal,
                    "attack_defense": defended_attack,
                },
                "attack_applied": {
                    name: bool(final[name].get("attack_applied", False))
                    for name in (
                        "attack",
                        "attack_defense_norm_clipping",
                        "attack_defense_coordinate_median",
                    )
                },
                "accepted_malicious_ratio": {
                    name: final[name].get("accepted_malicious_ratio", "N/A")
                    for name in (
                        "normal",
                        "attack",
                        "normal_defense_norm_clipping",
                        "attack_defense_norm_clipping",
                        "normal_defense_coordinate_median",
                        "attack_defense_coordinate_median",
                    )
                },
            }
        seed_rows.append({"seed": seed, "defenses": defense_results})

    means = {}
    for defense in ("norm_clipping", "coordinate_median"):
        means[defense] = {
            metric: sum(row["defenses"][defense][metric] for row in seed_rows) / len(seed_rows)
            for metric in (
                "attack_impact_percentage_points",
                "defense_improvement_percentage_points",
                "normal_defense_loss_percentage_points",
            )
        }
    report = {
        "comparison_id": prefix,
        "environment": {
            key: shared.get(key)
            for key in (
                "participation_rate",
                "delay_clients",
                "data_distribution",
                "num_rounds",
                "round_timeout",
                "model_config",
                "local_epochs",
                "batch_size",
                "learning_rate",
            )
        },
        "seeds": seed_rows,
        "mean": means,
        "run_result_paths": run_paths,
        "accuracy_unit": "fraction_0_to_1",
        "difference_unit": "percentage_points",
        "formulas": {
            "attack_impact": "normal_fedavg - attack_fedavg",
            "defense_improvement": "attack_plus_defense - attack_fedavg",
            "normal_defense_loss": "normal_fedavg - normal_plus_defense",
        },
        "comparison_path": str(report_path),
    }
    output.mkdir(parents=True, exist_ok=True)
    with report_path.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
    return report


def main():
    parser = argparse.ArgumentParser(description="정상·공격 학습을 순서대로 실제 실행하고 비교합니다.")
    parser.add_argument("experiment_id", help="새 비교 ID (예: compare_001)")
    parser.add_argument("--config", type=Path, help="매니저 설정을 덮어쓸 JSON 파일 (생략하면 현재 설정)")
    parser.add_argument("--manager-url", default="http://127.0.0.1:8000")
    parser.add_argument(
        "--security-seeds",
        help="모든 공격·방어 조합을 비교할 쉼표 구분 시드 (예: 42,43,44)",
    )
    args = parser.parse_args()
    try:
        config = json.loads(args.config.read_text(encoding="utf-8-sig")) if args.config else {}
        config["experiment_id"] = args.experiment_id
        print("정상 학습 → 공격 학습 → 비교를 시작합니다. 진행 상황은 매니저 웹 화면에서 확인하세요.", flush=True)
        if args.security_seeds:
            seeds = [int(value.strip()) for value in args.security_seeds.split(",")]
            result = run_full_security_comparison(
                config, seeds, manager_url=args.manager_url
            )
        else:
            result = run_comparison_experiment(config, manager_url=args.manager_url)
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    except (OSError, URLError, ValueError, RuntimeError, KeyError, TypeError) as error:
        parser.exit(2, f"오류: {error}\n")
    except KeyboardInterrupt:
        parser.exit(130, "비교 실행을 중단했습니다. 진행 중인 라운드의 종료 상태는 매니저에서 확인하세요.\n")


if __name__ == "__main__":
    main()
