import os
import random
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
import re
from common.training import TrainingSettings
from common.model import model_config, finite_number, positive_int

from exp_manager.services.experiment_output import save_experiment_outputs, save_round_result
from exp_manager.services.pretrained_model import parse_pretrained_global_model
from exp_manager.services.rest_client import RestClient
from exp_manager.utils.project_root_path import PROJECT_ROOT_PATH


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
        self.advanced_rounds = set()
        self.malicious_clients_by_round = {}
        self.dlg_images_by_round = {}
        self.dlg_results_by_round = {}
        self.dlg_participant_by_round = {}
        self.config = {
            **TrainingSettings().to_dict(),
            "model_config": model_config(),
            "round_timeout": 900.0,
            "experiment_id": f"fedsec-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            "experiment_type": "federated",
            "seed": 42,
            "num_rounds": 10,
            "dlg_training_rounds": 5,
            "use_pretrained_global_model": False,
            "pretrained_global_model": None,
            "pretrained_model_format": "json",
            "dlg_client_id": None,
            "participation_rate": 1.0,
            "malicious_clients": [0, 1],
            "attack_enabled": False,
            "attack_scale": -5.0,
            "round_type": "federated",
            "dlg_noise_sigma": 0.0,
            "dlg_defense_enabled": False,
            "dlg_iterations": 200,
            "dlg_image": [],
            "dlg_label": 0,
            "dlg_image_path": "",
            "dlg_label_path": "",
            "defense": "none",
            "defense_enabled": False,
            "clip_norm": 1.0,
            "delay_clients": [],
            "data_distribution": "iid",
            "round_configs": [],
        }
        env_config = self._load_dlg_env()
        self.config.update(env_config)
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

    def get_public_config(self) -> dict:
        config = self.get_config()
        pretrained_model = config.pop("pretrained_global_model", None)
        config["pretrained_global_model_loaded"] = pretrained_model is not None
        if pretrained_model:
            config["pretrained_global_model_version"] = pretrained_model.get(
                "model_version", 0
            )
        return config

    @staticmethod
    def _load_dlg_env() -> dict:
        env_path = PROJECT_ROOT_PATH / "exp_manager" / ".env"
        values = {}
        if env_path.exists():
            for raw_line in env_path.read_text(encoding="utf-8").splitlines():
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                values[key.strip()] = value.strip().strip("\"'")
        image_path = values.get("DLG_IMAGE_PATH", "")
        label_path = values.get("DLG_LABEL_PATH", "")
        return {
            "dlg_image_path": os.getenv("DLG_IMAGE_PATH", image_path),
            "dlg_label_path": os.getenv("DLG_LABEL_PATH", label_path),
            "dlg_label": int(os.getenv("DLG_LABEL", values.get("DLG_LABEL", "0"))),
        }

    def update_config(self, payload: dict) -> dict:
        with self.lock:
            if self.state["status"] in ("running", "stopping"):
                raise ValueError("실행 중에는 설정을 변경할 수 없습니다.")
            allowed = (
                "experiment_id",
                "experiment_type",
                "seed",
                "num_rounds",
                "dlg_training_rounds",
                "use_pretrained_global_model",
                "pretrained_model_format",
                "dlg_client_id",
                "participation_rate",
                "malicious_clients",
                "attack_enabled",
                "attack_scale",
                "round_type",
                "dlg_noise_sigma",
                "dlg_defense_enabled",
                "dlg_iterations",
                "dlg_image",
                "dlg_label",
                "dlg_image_path",
                "dlg_label_path",
                "defense",
                "defense_enabled",
                "clip_norm",
                "delay_clients",
                "data_distribution",
                "round_configs",
            )
            allowed = (*allowed, *TrainingSettings.__dataclass_fields__, "model_config", "round_timeout")
            previous = self.config
            self.config = {**previous, **{key: payload[key] for key in allowed if key in payload}}
            try:
                use_pretrained = bool(
                    payload.get(
                        "use_pretrained_global_model",
                        previous["use_pretrained_global_model"],
                    )
                )
                self.config["use_pretrained_global_model"] = use_pretrained
                model_data = payload.get("pretrained_model_data") or ""
                if not isinstance(model_data, str):
                    raise ValueError("전역 모델 데이터는 텍스트 형식이어야 합니다.")
                if not use_pretrained:
                    self.config["pretrained_global_model"] = None
                elif model_data.strip():
                    self.config["pretrained_global_model"] = (
                        parse_pretrained_global_model(
                            model_data,
                            payload.get(
                                "pretrained_model_format",
                                self.config["pretrained_model_format"],
                            ),
                            self.config["model_config"],
                        )
                    )
                elif previous.get("pretrained_global_model") is None:
                    raise ValueError("전역 모델 파일을 선택하거나 모델 값을 입력하세요.")
                self._validate_config()
            except (ValueError, TypeError, KeyError):
                self.config = previous
                raise
            return dict(self.config)

    def _validate_config(self):
        TrainingSettings.from_dict(self.config)
        model_config(self.config.get("model_config"))
        if self.config["pretrained_model_format"] not in {"json", "csv", "text"}:
            raise ValueError("지원하지 않는 전역 모델 입력 형식입니다.")
        if (
            self.config["use_pretrained_global_model"]
            and self.config["pretrained_global_model"] is None
        ):
            raise ValueError("DLG 실험에 사용할 사전 전역 모델이 등록되지 않았습니다.")
        finite_number(self.config["round_timeout"], "round_timeout", positive=True)
        positive_int(self.config["seed"], "seed", 0)
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,100}", self.config["experiment_id"]):
            raise ValueError("실험 ID는 영문·숫자·밑줄·하이픈으로 입력하세요.")
        if not 0 < self.config["participation_rate"] <= 1:
            raise ValueError("participation_rate는 0 초과 1 이하이어야 합니다.")
        if self.config["num_rounds"] < 1:
            raise ValueError("num_rounds는 1 이상이어야 합니다.")
        if self.config["dlg_training_rounds"] < 1:
            raise ValueError("DLG 사전 연합학습 라운드는 1 이상이어야 합니다.")
        dlg_client_id = self.config.get("dlg_client_id")
        if (
            dlg_client_id is not None
            and not 0 <= int(dlg_client_id) < self.client_count
        ):
            raise ValueError("DLG 참가 클라이언트 ID 범위가 올바르지 않습니다.")
        if self.config["defense"] not in {"none", "norm_clipping", "coordinate_median"}:
            raise ValueError("지원하지 않는 방어 방식입니다.")
        if float(self.config["dlg_noise_sigma"]) < 0:
            raise ValueError("DLG 가우시안 노이즈 sigma는 0 이상이어야 합니다.")
        malicious = self.config["malicious_clients"]
        if len(malicious) > self.client_count or len(set(malicious)) != len(malicious):
            raise ValueError("악성 클라이언트 설정이 올바르지 않습니다.")
        if any(
            client_id < 0 or client_id >= self.client_count for client_id in malicious
        ):
            raise ValueError("악성 클라이언트 ID 범위가 올바르지 않습니다.")
        if len(self.config["round_configs"]) > self.total_round_count():
            raise ValueError("라운드 설정 수가 전체 라운드 수보다 많습니다.")

    def total_round_count(self) -> int:
        if self.config.get("experiment_type") == "dlg":
            if self.config.get("use_pretrained_global_model"):
                return 1
            return int(self.config.get("dlg_training_rounds", 5)) + 1
        return int(self.config.get("num_rounds", 1))

    def status(self) -> dict:
        with self.lock:
            return {
                **self.state,
                "malicious_clients_by_round": dict(self.malicious_clients_by_round),
                "config": self.get_public_config(),
                "total_rounds": self.total_round_count(),
                "dlg_participant_by_round": dict(self.dlg_participant_by_round),
                "running": self.state["status"] == "running",
            }

    def start(self) -> dict:
        with self.lock:
            if self.state["status"] == "running":
                return self.status()
            if self.state["status"] == "stopping":
                raise RuntimeError("진행 중인 라운드가 종료될 때까지 기다려 주세요.")
            self._validate_config()
            directory = self.results_dir / self.config["experiment_id"]
            if (directory / "results.json").exists() or any((directory / "rounds").glob("*.json")):
                raise RuntimeError("결과가 이미 있는 실험 ID입니다. 새 ID를 입력하세요.")
            server_process = self.process_manager.server_process
            if not server_process or server_process.poll() is not None:
                raise RuntimeError("중앙 서버가 실행 중이어야 합니다.")
            self.process_manager.sync_client_registry()
            self.stop_event.clear()
            self.processed_rounds.clear()
            self.advanced_rounds.clear()
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
        if self.state["status"] in ("running", "stopping"):
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
            self.advanced_rounds.clear()
            self.dlg_participant_by_round.clear()
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
            if self.stop_event.is_set():
                with self.lock:
                    self.state.update(status="stopped", message="실험이 중단되었습니다.")
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
            timeout=float(round_config["round_timeout"]) + 120.0,
        )
        if result.get("round") == round_id:
            self.record_round_result(result, advance=True)

    def execute_dlg_round(self, round_id: int, round_config: dict) -> dict:
        """참가 클라이언트에서 실제 CIFAR-10 단일 샘플을 선택한다.

        서버에는 원본 이미지/레이블을 보내지 않는다. 원본은 클라이언트가
        /api/dlg/original 평가 관리자 endpoint로 직접 제출한다.
        """
        registry = self.process_manager.client_registry()
        if not registry:
            raise RuntimeError("DLG client is not registered")
        configured_client = self.config.get("dlg_client_id")
        if configured_client is None:
            rng = random.Random(int(self.config.get("seed", 42)) + round_id * 7919)
            client = rng.choice(registry)
        else:
            client = next(
                item for item in registry if item["client_id"] == int(configured_client)
            )
        self.dlg_participant_by_round[round_id] = client["client_id"]
        self.dlg_images_by_round[round_id] = []
        round_config["dlg_image_shape"] = [3, 32, 32]
        server_round_config = round_config
        if self.config.get("use_pretrained_global_model"):
            server_round_config = {
                **round_config,
                "initial_global_model": self.config["pretrained_global_model"],
            }
        self.rest_client.post_json(
            f"{self.process_manager.server_url()}/api/round/config",
            server_round_config,
        )
        return self.rest_client.post_json(
            f"{self.process_manager.server_url()}/api/dlg/participant",
            {
                **round_config,
                "client": client,
                "image_shape": [3, 32, 32],
                "noise_sigma": round_config.get("dlg_noise_sigma", 0.0),
                "defense_enabled": round_config.get("dlg_defense_enabled", False),
                "server_url": self.process_manager.server_url(),
            },
            timeout=float(round_config.get("round_timeout", 900)) + 120.0,
        )

    def store_dlg_original(self, round_id: int, experiment_id: str,
                           client_id: int, image: list[float]) -> None:
        """신뢰된 매니저에 원본 등록. 공격 서버에는 원본을 노출하지 않는다."""
        import math
        if experiment_id != self.config["experiment_id"]:
            raise ValueError("Wrong DLG experiment ID")
        if self.dlg_participant_by_round.get(round_id) != client_id:
            raise ValueError("Wrong DLG participant")
        if not isinstance(image, list) or len(image) != 3 * 32 * 32:
            raise ValueError("Expected 3072 CIFAR-10 pixels")
        if any(not isinstance(x, (int, float)) or not math.isfinite(x) or not 0 <= x <= 1 for x in image):
            raise ValueError("DLG original pixels must be finite in [0,1]")
        with self.lock:
            self.dlg_images_by_round[round_id] = list(image)

    def get_dlg_original(self, round_id: int) -> list[float]:
        return self.dlg_images_by_round.get(round_id, [])

    def record_dlg_result(self, round_id: int, result: dict) -> Path:
        """DLG MSE/이미지를 별도 저장하고, 실험 상태를 completed로 종료한다."""
        import json
        config = self.get_config()
        result = {
            **result,
            "experiment_id": config["experiment_id"],
            "round": round_id,
            "round_type": "dlg",
            "experiment_type": "dlg",
        }
        output_dir = self.results_dir / config["experiment_id"]
        output_dir.mkdir(parents=True, exist_ok=True)
        from exp_manager.services.dlg_report import save_dlg_images
        result["images"] = save_dlg_images(
            self.get_dlg_original(round_id), result["reconstructed_image"], output_dir
        )
        result_path = output_dir / "dlg_result.json"
        result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        with self.lock:
            self.dlg_results_by_round[round_id] = result
            self.state["current_round"] = round_id
            self.state["status"] = "completed"
            self.state["message"] = "실제 CNN DLG 복원 및 MSE 평가가 완료되었습니다."
            federated_results = list(self.state["round_results"])
        # 기존 accuracy/rounds 그래프는 정상 사전 학습 라운드만 사용한다.
        if federated_results:
            log_path, summary_path, chart_path = save_experiment_outputs(
                self.results_dir, config, federated_results
            )
            with self.lock:
                self.state.update(
                    log_path=str(log_path), result_path=str(summary_path), chart_path=str(chart_path)
                )
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

        rng = random.Random(int(experiment_config.get("seed", 42)) + round_id * 1009)
        candidates = list(round_config.get("malicious_clients", []))
        selected = sorted(rng.sample(candidates, min(2, len(candidates))))
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

    def record_round_result(self, result: dict, *, advance: bool = False) -> Path:
        """라운드 결과를 상태와 JSON 파일에 기록하고 예약된 다음 라운드를 시작한다."""
        round_id = int(result["round"])
        config = self.get_config()
        if result.get("experiment_id") != config["experiment_id"]:
            raise ValueError("다른 실험의 결과입니다.")
        round_config = self._round_config(round_id, config)
        malicious_clients = self.malicious_clients_by_round.get(
            round_id, result.get("malicious_clients", [])
        )
        attack_enabled = bool(round_config.get("attack_enabled", False))
        defense_enabled = bool(round_config.get("defense_enabled", False))
        result = {
            **result,
            "attack_enabled": bool(result.get("attack_enabled", attack_enabled)),
            "attack_type": (
                "insider_collusion"
                if result.get("attack_enabled", attack_enabled)
                else None
            ),
            "malicious_clients": malicious_clients,
            "malicious_selected": sorted(set(result.get("selected_clients", [])) & set(malicious_clients)),
            "defense_enabled": bool(result.get("defense_enabled", defense_enabled)),
            "defense": result.get("defense", round_config.get("defense", "none")),
            "defense_type": (
                result.get("defense")
                if result.get("defense_enabled", defense_enabled)
                else None
            ),
            "clip_norm": result.get("clip_norm", round_config.get("clip_norm", 1.0)),
            "attack_scale": result.get(
                "attack_scale", round_config.get("attack_scale", -5.0)
            ),
        }
        round_path = save_round_result(self.results_dir, config, result)

        with self.lock:
            should_advance = advance and round_id not in self.advanced_rounds
            if should_advance:
                self.advanced_rounds.add(round_id)
            self.processed_rounds.add(round_id)
            results = [
                item
                for item in self.state["round_results"]
                if item.get("round") != round_id
            ]
            results.append(result)
            results.sort(key=lambda item: item["round"])
            self.state["round_results"] = results
            self.state["current_round"] = round_id

        if (
            should_advance
            and round_id < self.total_round_count()
            and self.state["status"] == "running"
        ):
            next_round = round_id + 1
            threading.Thread(
                target=self._start_next_round,
                args=(config, next_round),
                daemon=True,
            ).start()
        elif (
            should_advance
            and round_id >= self.total_round_count()
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

    def _start_next_round(self, config: dict, round_id: int):
        try:
            self._start_round(config, round_id)
        except Exception as error:
            with self.lock:
                self.state.update(status="failed", message=f"실험 실패: {error}")

    def _round_config(self, round_id: int, config: dict) -> dict:
        """라운드별 설정이 있으면 기본 설정 위에 덮어쓴다."""
        custom_rounds = config.get("round_configs", [])
        custom = custom_rounds[round_id - 1] if round_id <= len(custom_rounds) else {}
        round_type = custom.get("round_type", "federated")
        if config.get("experiment_type") == "dlg":
            round_type = (
                "dlg"
                if round_id
                == (
                    1
                    if config.get("use_pretrained_global_model")
                    else int(config.get("dlg_training_rounds", 5)) + 1
                )
                else "federated"
            )
        return {
            **TrainingSettings.from_dict({**config, **custom}).to_dict(),
            "model_config": custom.get("model_config", config["model_config"]),
            "round_timeout": custom.get("round_timeout", config["round_timeout"]),
            "participation_rate": custom.get(
                "participation_rate", config["participation_rate"]
            ),
            "malicious_clients": custom.get(
                "malicious_clients", config["malicious_clients"]
            ),
            "attack_enabled": custom.get("attack_enabled", config["attack_enabled"]),
            "attack_scale": custom.get("attack_scale", config["attack_scale"]),
            "round_type": round_type,
            "dlg_noise_sigma": custom.get("dlg_noise_sigma", config["dlg_noise_sigma"]),
            "dlg_defense_enabled": custom.get(
                "dlg_defense_enabled", config["dlg_defense_enabled"]
            ),
            "dlg_iterations": custom.get("dlg_iterations", config["dlg_iterations"]),
            "dlg_image": custom.get("dlg_image", config["dlg_image"]),
            "dlg_label": custom.get("dlg_label", config["dlg_label"]),
            "dlg_image_path": custom.get("dlg_image_path", config["dlg_image_path"]),
            "dlg_label_path": custom.get("dlg_label_path", config["dlg_label_path"]),
            "defense": custom.get("defense", config["defense"]),
            "defense_enabled": custom.get("defense_enabled", config["defense_enabled"]),
            "clip_norm": custom.get("clip_norm", config["clip_norm"]),
            "delay_clients": custom.get("delay_clients", config["delay_clients"]),
            "data_distribution": custom.get(
                "data_distribution", config["data_distribution"]
            ),
        }

    def execute_single_round(self, round_id: int, payload: dict) -> dict:
        """관리자 요청을 클라이언트와 중앙 서버 REST API로 전달해 단일 라운드를 실행한다."""
        if self.state["status"] == "running":
            raise RuntimeError(
                "전체 실험 실행 중에는 단일 라운드를 실행할 수 없습니다."
            )
        if round_id < 1:
            raise ValueError("라운드 번호는 1 이상이어야 합니다.")

        config = self.get_config()
        round_config = dict(payload)
        round_config.setdefault("experiment_id", config["experiment_id"])
        round_config.setdefault("round", round_id)
        if round_config["experiment_id"] != config["experiment_id"]:
            raise ValueError("실험 ID 변경 후 설정을 먼저 저장하세요.")
        if (self.results_dir / config["experiment_id"] / "rounds" / f"round_{round_id:03d}.json").exists():
            raise RuntimeError("이미 저장된 라운드입니다. 새 실험 ID를 사용하세요.")

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
        if round_config.get("round_type") == "dlg":
            result = self.execute_dlg_round(round_id, round_config)
            with self.lock:
                self.state.update(
                    current_round=round_id,
                    status="completed",
                    message=f"{round_id}번 DLG 라운드를 시작했습니다.",
                )
            return result
        self.rest_client.post_json(
            f"{self.process_manager.server_url()}/api/round/config", round_config
        )

        result = self.rest_client.post_json(
            f"{self.process_manager.server_url()}/api/rounds/{round_id}/start",
            round_config,
            timeout=float(round_config["round_timeout"]) + 120.0,
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
