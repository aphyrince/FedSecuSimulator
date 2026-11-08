import random
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from services.experiment_output import save_experiment_outputs
from services.rest_client import RestClient


class ExperimentManager:
    def __init__(self, process_manager, results_dir: Path, client_count: int):
        self.process_manager = process_manager
        self.results_dir = results_dir
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.client_count = client_count
        self.lock = threading.RLock()
        self.worker = None
        self.stop_event = threading.Event()
        self.rest_client = RestClient()
        self.config = {
            "experiment_id": f"fedsec-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            "experiment_type": "federated",
            "seed": 42,
            "num_rounds": 10,
            "participation_rate": 0.5,
            "malicious_clients": [0, 1],
            "attack_enabled": False,
            "defense": "none",
            "clip_norm": 1.0,
            "delay_clients": [2, 5, 8],
            "data_distribution": "iid",
            "round_configs": [],
        }
        self.state = {
            "status": "idle",
            "current_round": 0,
            "round_results": [],
            "log_path": None,
            "result_path": None,
            "chart_path": None,
            "message": "",
        }

    def get_config(self) -> dict:
        with self.lock:
            return dict(self.config)

    def update_config(self, payload: dict) -> dict:
        with self.lock:
            if self.state["status"] == "running":
                raise ValueError("실행 중에는 설정을 변경할 수 없습니다.")
            allowed = (
                "experiment_id",
                "experiment_type",
                "seed",
                "num_rounds",
                "participation_rate",
                "malicious_clients",
                "attack_enabled",
                "defense",
                "clip_norm",
                "delay_clients",
                "data_distribution",
                "round_configs",
            )
            self.config.update({key: payload[key] for key in allowed if key in payload})
            self._validate_config()
            return dict(self.config)

    def _validate_config(self):
        if not 0 < self.config["participation_rate"] <= 1:
            raise ValueError("participation_rate는 0 초과 1 이하이어야 합니다.")
        if self.config["num_rounds"] < 1:
            raise ValueError("num_rounds는 1 이상이어야 합니다.")
        if self.config["defense"] not in {"none", "norm_clipping", "coordinate_median"}:
            raise ValueError("지원하지 않는 방어 방식입니다.")
        malicious = self.config["malicious_clients"]
        if len(malicious) > self.client_count or len(set(malicious)) != len(malicious):
            raise ValueError("악성 클라이언트 설정이 올바르지 않습니다.")
        if any(
            client_id < 0 or client_id >= self.client_count for client_id in malicious
        ):
            raise ValueError("악성 클라이언트 ID 범위가 올바르지 않습니다.")
        if len(self.config["round_configs"]) > self.config["num_rounds"]:
            raise ValueError("라운드 설정 수가 전체 라운드 수보다 많습니다.")

    def status(self) -> dict:
        with self.lock:
            return {
                **self.state,
                "config": dict(self.config),
                "running": self.state["status"] == "running",
            }

    def start(self) -> dict:
        with self.lock:
            if self.state["status"] == "running":
                return self.status()
            server_process = self.process_manager.server_process
            if not server_process or server_process.poll() is not None:
                raise RuntimeError("중앙 서버가 실행 중이어야 합니다.")
            self.process_manager.sync_client_registry()
            self.stop_event.clear()
            self.state.update(
                status="running",
                current_round=0,
                round_results=[],
                log_path=None,
                result_path=None,
                chart_path=None,
                message="실험을 시작했습니다.",
            )
            self.worker = threading.Thread(target=self._run, daemon=True)
            self.worker.start()
            return self.status()

    def stop(self) -> dict:
        self.stop_event.set()
        with self.lock:
            if self.state["status"] == "running":
                self.state["status"] = "stopping"
                self.state["message"] = "실험 중단 요청을 처리 중입니다."
        return self.status()

    def reset(self) -> dict:
        if self.state["status"] == "running":
            raise RuntimeError("실행 중인 실험은 먼저 중단해야 합니다.")
        with self.lock:
            self.state.update(
                status="idle",
                current_round=0,
                round_results=[],
                log_path=None,
                result_path=None,
                chart_path=None,
                message="실험 상태를 초기화했습니다.",
            )
        return self.status()

    def _run(self):
        config = self.get_config()
        rng = random.Random(config["seed"])
        round_results = []
        selected_count = max(1, round(self.client_count * config["participation_rate"]))
        try:
            for round_id in range(1, config["num_rounds"] + 1):
                if self.stop_event.is_set():
                    break
                round_config = self._round_config(round_id, config)
                round_config = {
                    **round_config,
                    "experiment_id": config["experiment_id"],
                    "round": round_id,
                    "seed": config["seed"],
                }
                self.rest_client.post_json(
                    f"{self.process_manager.server_url()}/api/round/config",
                    round_config,
                )
                result = self.rest_client.post_json(
                    f"{self.process_manager.server_url()}/api/rounds/{round_id}/start",
                    round_config,
                )
                round_results.append(result)
                with self.lock:
                    self.state["current_round"] = round_id
                    self.state["round_results"] = round_results.copy()
                time.sleep(0.08)

            if self.stop_event.is_set():
                with self.lock:
                    self.state["status"] = "stopped"
                    self.state["message"] = "실험이 중단되었습니다."
                return

            log_path, result_path, chart_path = save_experiment_outputs(
                self.results_dir, config, round_results
            )
            with self.lock:
                self.state.update(
                    status="completed",
                    log_path=str(log_path),
                    result_path=str(result_path),
                    chart_path=str(chart_path),
                    message="실험이 완료되었습니다.",
                )
        except Exception as exc:
            with self.lock:
                self.state["status"] = "failed"
                self.state["message"] = f"실험 실패: {exc}"

    def _round_config(self, round_id: int, config: dict) -> dict:
        """라운드별 설정이 있으면 기본 설정 위에 덮어쓴다."""
        custom_rounds = config.get("round_configs", [])
        custom = custom_rounds[round_id - 1] if round_id <= len(custom_rounds) else {}
        return {
            "participation_rate": custom.get("participation_rate", config["participation_rate"]),
            "malicious_clients": custom.get("malicious_clients", config["malicious_clients"]),
            "attack_enabled": custom.get("attack_enabled", config["attack_enabled"]),
            "defense": custom.get("defense", config["defense"]),
            "clip_norm": custom.get("clip_norm", config["clip_norm"]),
            "delay_clients": custom.get("delay_clients", config["delay_clients"]),
            "data_distribution": custom.get("data_distribution", config["data_distribution"]),
        }

    def execute_single_round(self, round_id: int, payload: dict) -> dict:
        """관리자 요청을 클라이언트와 중앙 서버 REST API로 전달해 단일 라운드를 실행한다."""
        if self.state["status"] == "running":
            raise RuntimeError("전체 실험 실행 중에는 단일 라운드를 실행할 수 없습니다.")
        if round_id < 1:
            raise ValueError("라운드 번호는 1 이상이어야 합니다.")

        config = self.get_config()
        round_config = dict(payload)
        round_config.setdefault("experiment_id", config["experiment_id"])
        round_config.setdefault("round", round_id)

        inherited = self._round_config(round_id, config)
        round_config = {**inherited, **round_config}
        round_config["seed"] = config["seed"]

        # 중앙 서버에 먼저 라운드 설정을 전달한다.
        self.rest_client.post_json(
            f"{self.process_manager.server_url()}/api/round/config", round_config
        )

        result = self.rest_client.post_json(
            f"{self.process_manager.server_url()}/api/rounds/{round_id}/start",
            round_config,
        )
        with self.lock:
            results = [
                item
                for item in self.state["round_results"]
                if item.get("round") != round_id
            ]
            results.append(result)
            results.sort(key=lambda item: item["round"])
            self.state.update(
                round_results=results,
                current_round=round_id,
                status="completed",
                message=f"{round_id}번 라운드가 완료되었습니다.",
            )
        return result
