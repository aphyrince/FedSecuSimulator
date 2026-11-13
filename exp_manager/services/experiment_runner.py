import os
import random
import json
import threading
import time
import uuid
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
import re
from common.training import TrainingSettings
from common.model import build_model, model_config, finite_number, positive_int

from exp_manager.services.experiment_output import save_experiment_outputs, save_round_result
from exp_manager.services.pretrained_model import parse_pretrained_global_model
from exp_manager.services.rest_client import RestClient
from exp_manager.utils.project_root_path import PROJECT_ROOT_PATH


CONTINUOUS_ENVIRONMENTS = {
    "baseline": {
        "label": "기준",
        "participation_rate": 1.0,
        "delay_clients": [],
        "data_distribution": "iid",
    },
    "delay": {
        "label": "지연",
        "participation_rate": 1.0,
        "delay_clients": [2, 5, 8],
        "data_distribution": "iid",
    },
    "partial": {
        "label": "부분 참여",
        "participation_rate": 0.5,
        "delay_clients": [],
        "data_distribution": "iid",
    },
    "label-bias": {
        "label": "레이블 편향",
        "participation_rate": 1.0,
        "delay_clients": [],
        "data_distribution": "non_iid",
    },
}

CONTINUOUS_SECURITY_SCENARIOS = {
    "normal": {
        "label": "정상",
        "attack_enabled": False,
        "defense_enabled": False,
    },
    "attack": {
        "label": "공격",
        "attack_enabled": True,
        "defense_enabled": False,
    },
    "defense": {
        "label": "방어",
        "attack_enabled": False,
        "defense_enabled": True,
    },
    "attack_defense": {
        "label": "공격 + 방어",
        "attack_enabled": True,
        "defense_enabled": True,
    },
}


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
            "global_model_export_path": None,
            "continuous_experiment": None,
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
            if self.state["status"] in ("running", "stopping") or self._continuous_is_active():
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
        finite_number(self.config["attack_scale"], "attack_scale")
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
                "running": (
                    self.state["status"] == "running" or self._continuous_is_active()
                ),
            }

    def _continuous_is_active(self) -> bool:
        batch = self.state.get("continuous_experiment")
        return bool(batch and batch.get("status") in {"running", "stopping"})

    def start(self) -> dict:
        with self.lock:
            if self._continuous_is_active():
                raise RuntimeError("연속 실험이 진행 중입니다.")
            if self.state["status"] == "running":
                return self.status()
            if self.state["status"] == "stopping" or self._continuous_is_active():
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
                global_model_export_path=None,
                continuous_experiment=None,
                message="실험을 시작했습니다.",
            )
            self.worker = threading.Thread(target=self._run, daemon=True)
            self.worker.start()
            return self.status()

    def start_continuous_experiments(self, cases: list[dict]) -> dict:
        """16개 내부자 공격 조합을 별도 ID로 순차 실행한다."""
        with self.lock:
            if self.state["status"] in ("running", "stopping") or self._continuous_is_active():
                raise RuntimeError("다른 실험이 실행 중입니다.")
            if self.config.get("experiment_type") != "federated":
                raise ValueError("연속 실험은 내부자 협력공격 모드에서만 사용할 수 있습니다.")
            self._validate_config()
            server_process = self.process_manager.server_process
            if not server_process or server_process.poll() is not None:
                raise RuntimeError("중앙 서버가 실행 중이어야 합니다.")

            batch_id = (
                f"batch-{datetime.now().strftime('%Y%m%d-%H%M%S')}-"
                f"{uuid.uuid4().hex[:6]}"
            )
            base_config = deepcopy(self.config)
            prepared_cases = self._prepare_continuous_cases(
                cases, batch_id, base_config
            )
            self.process_manager.sync_client_registry()

            batch_dir = self.results_dir / "continuous_runs" / batch_id
            batch_dir.mkdir(parents=True, exist_ok=False)
            manifest_path = batch_dir / "manifest.json"
            manifest = {
                "batch_id": batch_id,
                "experiment_type": "federated",
                "status": "running",
                "total_cases": len(prepared_cases),
                "current_case": 0,
                "started_at": datetime.now(timezone.utc).isoformat(),
                "completed_at": None,
                "cases": [
                    self._continuous_case_manifest(index, config, "pending")
                    for index, config in enumerate(prepared_cases, start=1)
                ],
            }
            self._write_continuous_manifest(manifest_path, manifest)
            for index, config in enumerate(prepared_cases, start=1):
                self._write_continuous_experiment_info(index, config, len(prepared_cases))

            self.stop_event.clear()
            self.state.update(
                status="running",
                current_round=0,
                round_results=[],
                log_path=None,
                result_path=None,
                chart_path=None,
                global_model_export_path=None,
                continuous_experiment={
                    "batch_id": batch_id,
                    "status": "running",
                    "current_case": 0,
                    "total_cases": len(prepared_cases),
                    "current_case_name": "",
                    "manifest_path": str(manifest_path),
                    "cases": deepcopy(manifest["cases"]),
                },
                message=f"연속 실험 {len(prepared_cases)}개를 시작했습니다.",
            )
            self.worker = threading.Thread(
                target=self._run_continuous_experiments,
                args=(prepared_cases, manifest, manifest_path, base_config),
                daemon=True,
            )
            self.worker.start()
            return self.status()

    def _prepare_continuous_cases(
        self, cases: list[dict], batch_id: str, base_config: dict
    ) -> list[dict]:
        if not isinstance(cases, list) or len(cases) != 16:
            raise ValueError("연속 실험은 16개 조합을 모두 포함해야 합니다.")
        expected_pairs = {
            (environment, security)
            for environment in CONTINUOUS_ENVIRONMENTS
            for security in CONTINUOUS_SECURITY_SCENARIOS
        }
        prepared = []
        seen_pairs = set()
        for index, item in enumerate(cases, start=1):
            if not isinstance(item, dict):
                raise ValueError(f"{index}번 연속 실험 설정 형식이 올바르지 않습니다.")
            environment_key = item.get("environment_template")
            security_key = item.get("security_scenario")
            pair = (environment_key, security_key)
            if pair not in expected_pairs or pair in seen_pairs:
                raise ValueError("연속 실험은 환경·보안 조합 16개를 한 번씩 포함해야 합니다.")
            seen_pairs.add(pair)
            environment = CONTINUOUS_ENVIRONMENTS[environment_key]
            security = CONTINUOUS_SECURITY_SCENARIOS[security_key]
            name = item.get(
                "name", f"{environment['label']} · {security['label']}"
            )
            if not isinstance(name, str) or not name.strip() or len(name) > 80:
                raise ValueError(f"{index}번 실험 이름은 1~80자로 입력하세요.")

            participation_rate = finite_number(
                item.get("participation_rate", environment["participation_rate"]),
                "participation_rate",
                positive=True,
            )
            if participation_rate > 1:
                raise ValueError("참여율은 0 초과 1 이하여야 합니다.")
            delay_clients = item.get("delay_clients", environment["delay_clients"])
            if (
                not isinstance(delay_clients, list)
                or len(set(delay_clients)) != len(delay_clients)
                or any(
                    type(client_id) is not int
                    or not 0 <= client_id < self.client_count
                    for client_id in delay_clients
                )
            ):
                raise ValueError(f"{index}번 지연 클라이언트 설정이 올바르지 않습니다.")
            distribution = item.get("data_distribution", environment["data_distribution"])
            if distribution not in {"iid", "non_iid"}:
                raise ValueError(f"{index}번 데이터 분포 설정이 올바르지 않습니다.")
            attack_enabled = item.get("attack_enabled", security["attack_enabled"])
            defense_enabled = item.get("defense_enabled", security["defense_enabled"])
            if not isinstance(attack_enabled, bool) or not isinstance(defense_enabled, bool):
                raise ValueError(f"{index}번 공격·방어 설정은 참/거짓이어야 합니다.")
            default_defense = base_config.get("defense", "norm_clipping")
            if default_defense == "none":
                default_defense = "norm_clipping"
            defense = item.get("defense", default_defense)
            if defense not in {"norm_clipping", "coordinate_median"}:
                raise ValueError(f"{index}번 방어 방식이 올바르지 않습니다.")
            attack_scale = finite_number(
                item.get("attack_scale", base_config["attack_scale"]), "attack_scale"
            )
            num_rounds = positive_int(
                item.get("num_rounds", base_config["num_rounds"]), "num_rounds"
            )

            experiment_id = (
                f"{batch_id}-{environment_key}-{security_key}"
            )
            if len(experiment_id) > 100:
                raise ValueError("자동 생성된 실험 ID가 허용 길이를 초과했습니다.")
            output_dir = self.results_dir / experiment_id
            if output_dir.exists():
                raise RuntimeError(f"실험 ID 결과 폴더가 이미 존재합니다: {experiment_id}")

            config = {
                **base_config,
                "experiment_id": experiment_id,
                "experiment_type": "federated",
                "num_rounds": num_rounds,
                "round_configs": [],
                "participation_rate": participation_rate,
                "delay_clients": delay_clients,
                "data_distribution": distribution,
                "attack_enabled": attack_enabled,
                "attack_scale": attack_scale,
                "defense_enabled": defense_enabled,
                "defense": defense,
                "continuous_batch_id": batch_id,
                "continuous_case_index": index,
                "continuous_case_count": 16,
                "continuous_scenario": {
                    "name": name.strip(),
                    "environment_template": environment_key,
                    "environment_label": environment["label"],
                    "security_scenario": security_key,
                    "security_label": security["label"],
                },
            }
            self._validate_config_for_continuous_case(config, index)
            prepared.append(config)
        if seen_pairs != expected_pairs:
            raise ValueError("연속 실험에서 환경·보안 조합이 누락되었습니다.")
        return prepared

    def _validate_config_for_continuous_case(self, config: dict, index: int):
        TrainingSettings.from_dict(config)
        finite_number(config["round_timeout"], f"{index}번 round_timeout", positive=True)
        positive_int(config["seed"], f"{index}번 seed", 0)

    @staticmethod
    def _continuous_case_manifest(index: int, config: dict, status: str) -> dict:
        scenario = config["continuous_scenario"]
        return {
            "case_index": index,
            "experiment_id": config["experiment_id"],
            "name": scenario["name"],
            "environment_template": scenario["environment_template"],
            "environment_label": scenario["environment_label"],
            "security_scenario": scenario["security_scenario"],
            "security_label": scenario["security_label"],
            "status": status,
            "result_directory": config["experiment_id"],
            "configuration": {
                key: config[key]
                for key in (
                    "num_rounds",
                    "participation_rate",
                    "delay_clients",
                    "data_distribution",
                    "attack_enabled",
                    "attack_scale",
                    "defense_enabled",
                    "defense",
                )
            },
        }

    def _write_continuous_experiment_info(
        self, index: int, config: dict, total_cases: int
    ):
        experiment_dir = self.results_dir / config["experiment_id"]
        experiment_dir.mkdir(parents=True, exist_ok=False)
        info = {
            "experiment_id": config["experiment_id"],
            "experiment_type": "federated",
            "batch_id": config["continuous_batch_id"],
            "case_index": index,
            "case_count": total_cases,
            "scenario": config["continuous_scenario"],
            "configuration": config,
        }
        (experiment_dir / "experiment_info.json").write_text(
            json.dumps(info, ensure_ascii=False, indent=2, allow_nan=False),
            encoding="utf-8",
        )

    @staticmethod
    def _write_continuous_manifest(path: Path, manifest: dict):
        temporary_path = path.with_suffix(".json.tmp")
        temporary_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False),
            encoding="utf-8",
        )
        temporary_path.replace(path)

    def _run_continuous_experiments(
        self,
        cases: list[dict],
        manifest: dict,
        manifest_path: Path,
        base_config: dict,
    ):
        try:
            for index, config in enumerate(cases, start=1):
                if self.stop_event.is_set():
                    break
                with self.lock:
                    self.config = deepcopy(config)
                    self.processed_rounds.clear()
                    self.advanced_rounds.clear()
                    self.malicious_clients_by_round.clear()
                    self.dlg_participant_by_round.clear()
                    self.state.update(
                        status="running",
                        current_round=0,
                        round_results=[],
                        log_path=None,
                        result_path=None,
                        chart_path=None,
                        global_model_export_path=None,
                        message=f"연속 실험 {index}/16: {config['continuous_scenario']['name']}",
                    )
                    batch_state = self.state["continuous_experiment"]
                    batch_state.update(
                        status="running",
                        current_case=index,
                        current_case_name=config["continuous_scenario"]["name"],
                    )
                    batch_state["cases"][index - 1]["status"] = "running"
                manifest["current_case"] = index
                manifest["cases"][index - 1]["status"] = "running"
                self._write_continuous_manifest(manifest_path, manifest)

                self._start_round(config, 1)
                while True:
                    with self.lock:
                        case_status = self.state["status"]
                    if case_status in {"completed", "failed", "stopped"}:
                        break
                    if self.stop_event.is_set():
                        break
                    time.sleep(0.1)

                with self.lock:
                    case_status = self.state["status"]
                    case_message = self.state["message"]
                case_manifest = manifest["cases"][index - 1]
                case_manifest["status"] = (
                    "stopped" if self.stop_event.is_set() else case_status
                )
                case_manifest["message"] = case_message
                case_manifest["result_path"] = str(
                    self.results_dir / config["experiment_id"] / "results.json"
                )
                with self.lock:
                    self.state["continuous_experiment"]["cases"][index - 1] = deepcopy(
                        case_manifest
                    )
                self._write_continuous_manifest(manifest_path, manifest)
                if self.stop_event.is_set() or case_status != "completed":
                    break

            stopped = self.stop_event.is_set()
            failed = any(case["status"] == "failed" for case in manifest["cases"])
            final_status = "stopped" if stopped else "failed" if failed else "completed"
            manifest["status"] = final_status
            manifest["completed_at"] = datetime.now(timezone.utc).isoformat()
            self._write_continuous_manifest(manifest_path, manifest)
            with self.lock:
                self.config = deepcopy(base_config)
                self.state.update(
                    status=final_status,
                    message=(
                        "연속 실험이 중단되었습니다."
                        if stopped
                        else "연속 실험이 실패했습니다."
                        if failed
                        else "연속 실험 16개가 모두 완료되었습니다."
                    ),
                )
                batch_state = self.state["continuous_experiment"]
                batch_state["status"] = final_status
                batch_state["cases"] = deepcopy(manifest["cases"])
        except Exception as exc:
            for case in manifest["cases"]:
                if case["status"] == "running":
                    case["status"] = "failed"
                    case["message"] = str(exc)
                    break
            manifest["status"] = "failed"
            manifest["completed_at"] = datetime.now(timezone.utc).isoformat()
            manifest["error"] = str(exc)
            self._write_continuous_manifest(manifest_path, manifest)
            with self.lock:
                self.config = deepcopy(base_config)
                self.state.update(status="failed", message=f"연속 실험 실패: {exc}")
                batch_state = self.state.get("continuous_experiment")
                if batch_state:
                    batch_state["status"] = "failed"
                    batch_state["cases"] = deepcopy(manifest["cases"])

    def stop(self) -> dict:
        self.stop_event.set()
        with self.lock:
            if self.state["status"] == "running" or self._continuous_is_active():
                self.state["status"] = "stopping"
                self.state["message"] = "실험 중단 요청을 처리 중입니다."
                batch = self.state.get("continuous_experiment")
                if batch and batch["status"] == "running":
                    batch["status"] = "stopping"
        return self.status()

    def reset(self) -> dict:
        if self.state["status"] in ("running", "stopping") or self._continuous_is_active():
            raise RuntimeError("실행 중인 실험은 먼저 중단해야 합니다.")
        with self.lock:
            self.state.update(
                status="idle",
                current_round=0,
                round_results=[],
                log_path=None,
                result_path=None,
                chart_path=None,
                global_model_export_path=None,
                continuous_experiment=None,
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
                    "attack_scale": round_config.get("attack_scale", -5.0),
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
            export_path = None
            if config.get("experiment_type") == "federated":
                export_path = self._save_final_global_model_for_dlg(config)
            log_path, result_path, chart_path = save_experiment_outputs(
                self.results_dir, config, self.state["round_results"]
            )
            with self.lock:
                self.state.update(
                    status="completed",
                    log_path=str(log_path),
                    result_path=str(result_path),
                    chart_path=str(chart_path),
                    global_model_export_path=(
                        str(export_path) if export_path is not None else None
                    ),
                    message=(
                        f"실험이 완료되었습니다. DLG용 전역 모델 저장: {export_path}"
                        if export_path is not None
                        else "실험이 완료되었습니다."
                    ),
                )
        return round_path

    def _save_final_global_model_for_dlg(self, config: dict) -> Path:
        """Export the final server model as a JSON checkpoint accepted by DLG settings."""
        model = self.rest_client.get_json(
            f"{self.process_manager.server_url()}/api/model/global",
            timeout=60.0,
        )
        if (
            not isinstance(model, dict)
            or model.get("experiment_id") != config["experiment_id"]
        ):
            raise RuntimeError(
                "중앙 서버에서 현재 실험의 최종 전역 모델을 가져오지 못했습니다."
            )
        build_model(model)

        export_path = (
            self.results_dir / config["experiment_id"] / "global_model_for_dlg.json"
        )
        export_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = export_path.with_suffix(".json.tmp")
        temporary_path.write_text(
            json.dumps(model, ensure_ascii=False, indent=2, allow_nan=False),
            encoding="utf-8",
        )
        temporary_path.replace(export_path)
        return export_path

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
        if self.state["status"] == "running" or self._continuous_is_active():
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
