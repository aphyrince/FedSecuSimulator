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


def main():
    parser = argparse.ArgumentParser(description="정상·공격 학습을 순서대로 실제 실행하고 비교합니다.")
    parser.add_argument("experiment_id", help="새 비교 ID (예: compare_001)")
    parser.add_argument("--config", type=Path, help="매니저 설정을 덮어쓸 JSON 파일 (생략하면 현재 설정)")
    parser.add_argument("--manager-url", default="http://127.0.0.1:8000")
    args = parser.parse_args()
    try:
        config = json.loads(args.config.read_text(encoding="utf-8-sig")) if args.config else {}
        config["experiment_id"] = args.experiment_id
        print("정상 학습 → 공격 학습 → 비교를 시작합니다. 진행 상황은 매니저 웹 화면에서 확인하세요.", flush=True)
        result = run_comparison_experiment(config, manager_url=args.manager_url)
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    except (OSError, URLError, ValueError, RuntimeError, KeyError, TypeError) as error:
        parser.exit(2, f"오류: {error}\n")
    except KeyboardInterrupt:
        parser.exit(130, "비교 실행을 중단했습니다. 진행 중인 라운드의 종료 상태는 매니저에서 확인하세요.\n")


if __name__ == "__main__":
    main()
