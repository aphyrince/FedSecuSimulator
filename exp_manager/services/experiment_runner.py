import random
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from services.experiment_output import save_experiment_outputs, save_round_result
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
        self.processed_rounds = set()
        self.malicious_clients_by_round = {}
        self.dlg_images_by_round = {}
        self.dlg_results_by_round = {}
        self.config = {
            "experiment_id": f"fedsec-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            "experiment_type": "federated",
            "seed": 42,
            "num_rounds": 10,
            "participation_rate": 0.5,
            "malicious_clients": [0, 1],
            "attack_enabled": False,
            "attack_scale": -5.0,
            "round_type": "federated",
            "dlg_noise_sigma": 0.0,
            "dlg_iterations": 200,
            "dlg_image": [],
            "dlg_label": 0,
            "defense": "none",
            "defense_enabled": False,
            "clip_norm": 1.0,
            "delay_clients": [2, 5, 8],
            "data_distribution": "iid",
            "round_configs": [],
        }
        self.state = {
            "status": "idle",
            "current_round": 0,
            "round_results": [],
            "malicious_clients_by_round": {},
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
                "attack_scale",
                "round_type",
                "dlg_noise_sigma",
                "dlg_iterations",
                "dlg_image",
                "dlg_label",
                "defense",
                "defense_enabled",
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
                "malicious_clients_by_round": dict(self.malicious_clients_by_round),
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
            self.processed_rounds.clear()
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
            self.malicious_clients_by_round.clear()
            self.processed_rounds.clear()
        return self.status()

    def _run(self):
        config = self.get_config()
        round_results = []
        try:
            if self.stop_event.is_set():
                with self.lock:
                    self.state["status"] = "stopped"
                    self.state["message"] = "실험이 중단되었습니다."
                return
            self._start_round(config, 1)
        except Exception as exc:
            with self.lock:
                self.state["status"] = "failed"
                self.state["message"] = f"실험 실패: {exc}"

    def _start_round(self, config: dict, round_id: int):
        """라운드 설정을 중앙 서버에 전달하고 개별 라운드 시작 API를 호출한다."""
        if self.stop_event.is_set():
            return
        round_config = {
            **self._round_config(round_id, config),
            "experiment_id": config["experiment_id"],
            "round": round_id,
            "seed": config["seed"],
            "nums_clients": self.client_count,
            "experiment_manager_url": self.process_manager.manager_url(),
        }
        malicious_clients = self.prepare_malicious_clients(
            round_id, round_config, config
        )
        round_config["malicious_clients"] = malicious_clients
        if round_config.get("round_type") == "dlg":
            self.execute_dlg_round(round_id, round_config)
            return
        self.rest_client.post_json(
            f"{self.process_manager.server_url()}/api/round/config",
            round_config,
        )
        result = self.rest_client.post_json(
            f"{self.process_manager.server_url()}/api/rounds/{round_id}/start",
            round_config,
        )
        if result.get("round") == round_id:
            self.record_round_result(result)

    def execute_dlg_round(self, round_id: int, round_config: dict) -> dict:
        """DLG 체크포인트와 원본 이미지를 대상 클라이언트에 전달한다."""
        image = round_config.get("dlg_image", [])
        self.dlg_images_by_round[round_id] = list(image)
        registry = self.process_manager.client_registry()
        if not registry:
            raise RuntimeError("DLG client is not registered")
        client = registry[0]
        checkpoint = self.rest_client.get_json(
            f"{self.process_manager.server_url()}/api/model/global"
        )
        result = self.rest_client.post_json(
            f"{client['url']}/api/dlg/start",
            {
                **round_config,
                "global_model": checkpoint,
                "image": image,
                "image_shape": round_config.get("dlg_image_shape", [len(image)]),
                "noise_sigma": round_config.get("dlg_noise_sigma", 0.0),
                "server_url": self.process_manager.server_url(),
                "model_version": checkpoint.get("model_version", 0),
            },
        )
        return result

    def get_dlg_original(self, round_id: int) -> list[float]:
        return self.dlg_images_by_round.get(round_id, [])

    def record_dlg_result(self, round_id: int, result: dict) -> Path:
        result_path = save_round_result(self.results_dir, self.get_config(), result)
        with self.lock:
            self.dlg_results_by_round[round_id] = result
        return result_path

    def prepare_malicious_clients(
        self, round_id: int, round_config: dict, experiment_config: dict
    ) -> list[int]:
        """공격 라운드에서 악성 클라이언트 2개를 선택하고 피어 정보를 전달한다."""
        attack_enabled = bool(round_config.get("attack_enabled", False))
        if not attack_enabled:
            with self.lock:
                self.malicious_clients_by_round[round_id] = []
            return []

        rng = random.Random(
            int(experiment_config.get("seed", 42)) + round_id * 1009
        )
        selected = sorted(rng.sample(range(self.client_count), 2))
        with self.lock:
            self.malicious_clients_by_round[round_id] = selected
            self.state["malicious_clients_by_round"] = dict(
                self.malicious_clients_by_round
            )

        registry = {
            client["client_id"]: client
            for client in self.process_manager.client_registry()
        }
        for client_id in selected:
            peers = [
                {
                    "client_id": peer_id,
                    "ip": registry[peer_id]["ip"],
                    "port": registry[peer_id]["port"],
                    "url": registry[peer_id]["url"],
                }
                for peer_id in selected
                if peer_id != client_id
            ]
            self.rest_client.post_json(
                f"{registry[client_id]['url']}/api/round/malicious-peers",
                {
                    "round": round_id,
                    "malicious": True,
                    "peers": peers,
                },
            )
        return selected

    def record_round_result(self, result: dict) -> Path:
        """라운드 결과를 상태와 JSON 파일에 기록하고 예약된 다음 라운드를 시작한다."""
        round_id = int(result["round"])
        config = self.get_config()
        round_config = self._round_config(round_id, config)
        malicious_clients = self.malicious_clients_by_round.get(
            round_id, result.get("malicious_clients", [])
        )
        attack_enabled = bool(round_config.get("attack_enabled", False))
        defense_enabled = bool(
            round_config.get("defense_enabled", False)
        )
        result = {
            **result,
            "attack_enabled": bool(result.get("attack_enabled", attack_enabled)),
            "attack_type": (
                "insider_collusion"
                if result.get("attack_enabled", attack_enabled)
                else None
            ),
            "malicious_clients": malicious_clients,
            "malicious_selected": malicious_clients,
            "defense_enabled": bool(
                result.get("defense_enabled", defense_enabled)
            ),
            "defense": result.get(
                "defense", round_config.get("defense", "none")
            ),
            "defense_type": (
                result.get("defense")
                if result.get("defense_enabled", defense_enabled)
                else None
            ),
            "clip_norm": result.get(
                "clip_norm", round_config.get("clip_norm", 1.0)
            ),
            "attack_scale": result.get(
                "attack_scale", round_config.get("attack_scale", -5.0)
            ),
        }
        round_path = save_round_result(self.results_dir, config, result)

        with self.lock:
            is_duplicate = round_id in self.processed_rounds
            self.processed_rounds.add(round_id)
            results = [
                item for item in self.state["round_results"]
                if item.get("round") != round_id
            ]
            results.append(result)
            results.sort(key=lambda item: item["round"])
            self.state["round_results"] = results
            self.state["current_round"] = round_id

        if (
            not is_duplicate
            and round_id < config["num_rounds"]
            and self.state["status"] == "running"
        ):
            next_round = round_id + 1
            threading.Thread(
                target=self._start_round,
                args=(config, next_round),
                daemon=True,
            ).start()
        elif (
            not is_duplicate
            and round_id >= config["num_rounds"]
            and self.state["status"] == "running"
        ):
            log_path, result_path, chart_path = save_experiment_outputs(
                self.results_dir, config, self.state["round_results"]
            )
            with self.lock:
                self.state.update(
                    status="completed",
                    log_path=str(log_path),
                    result_path=str(result_path),
                    chart_path=str(chart_path),
                    message="실험이 완료되었습니다.",
                )
        return round_path

    def _round_config(self, round_id: int, config: dict) -> dict:
        """라운드별 설정이 있으면 기본 설정 위에 덮어쓴다."""
        custom_rounds = config.get("round_configs", [])
        custom = custom_rounds[round_id - 1] if round_id <= len(custom_rounds) else {}
        return {
            "participation_rate": custom.get("participation_rate", config["participation_rate"]),
            "malicious_clients": custom.get("malicious_clients", config["malicious_clients"]),
            "attack_enabled": custom.get("attack_enabled", config["attack_enabled"]),
            "attack_scale": custom.get("attack_scale", config["attack_scale"]),
            "round_type": custom.get("round_type", "federated"),
            "dlg_noise_sigma": custom.get(
                "dlg_noise_sigma", config["dlg_noise_sigma"]
            ),
            "dlg_iterations": custom.get(
                "dlg_iterations", config["dlg_iterations"]
            ),
            "dlg_image": custom.get("dlg_image", config["dlg_image"]),
            "dlg_label": custom.get("dlg_label", config["dlg_label"]),
            "defense": custom.get("defense", config["defense"]),
            "defense_enabled": custom.get(
                "defense_enabled", config["defense_enabled"]
            ),
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
        round_config["nums_clients"] = self.client_count
        round_config.setdefault(
            "experiment_manager_url", self.process_manager.manager_url()
        )

        # 중앙 서버에 먼저 라운드 설정을 전달한다.
        malicious_clients = self.prepare_malicious_clients(
            round_id, round_config, config
        )
        round_config["malicious_clients"] = malicious_clients
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
