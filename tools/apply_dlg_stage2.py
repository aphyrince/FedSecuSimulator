"""FedSecuSimulator feature/security-integration DLG stage2 REST patch.

현재 GitHub 공개 소스의 명확한 앵커를 검사한 뒤 변경. 원본은 results/dlg_stage2_backup에 저장.
원본과 다르면 중단하고 파일 변경을 하지 않는다.
"""
from __future__ import annotations
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BACKUP = ROOT / 'results' / 'dlg_stage2_backup'


def apply_replacements(text: str, changes: list[tuple[str, str]], path: str) -> str:
    for before, after in changes:
        if after and after in text:
            continue
        if not after and before not in text:
            continue
        if text.count(before) != 1:
            raise RuntimeError(f'{path}: 예상한 코드 위치를 찾지 못했습니다. 기존 파일이 수정됐을 수 있습니다.\n찾는 코드: {before[:110]!r}')
        text = text.replace(before, after, 1)
    return text


changes: dict[str, list[tuple[str, str]]] = {}

# 1. 클라이언트: 실제 이미지 1장 = 할당된 CIFAR-10 데이터에서 읽기.
# 원본은 평가용 매니저에만 보내며, 중앙 서버에는 노이즈가 들어간 그래디언트만 보낸다.
client_real = '''    if payload.get("global_model", {}).get("architecture") == MODEL_FORMAT:
        from fedsec.data import CifarRangeDataset
        from security.dlg_api import make_gradient_payload, original_pixels

        state = current_app.extensions["client_state"]
        try:
            sample, label = CifarRangeDataset(state.dataset_range, limit=1)[0]
            sigma = float(payload.get("noise_sigma", 0.0)) if payload.get("defense_enabled") else 0.0
            gradient = make_gradient_payload(
                payload["global_model"], sample, label,
                sigma=sigma,
                seed=int(payload.get("seed", 42)) + int(payload.get("round", 1)) + current_app.config["CLIENT_ID"],
            )
        except (ValueError, TypeError, OSError, RuntimeError) as exc:
            return jsonify(error=f"CIFAR-10 DLG gradient failed: {exc}"), 400

        manager_url = payload.get("experiment_manager_url")
        if not manager_url:
            return jsonify(error="experiment_manager_url is required for DLG evaluation"), 400
        round_id = int(payload.get("round", 1))
        client_id = current_app.config["CLIENT_ID"]
        # 이 경로는 실험 관리자에게만 원본을 전달한다. 중앙 공격 서버는 원본/라벨을 못 본다.
        post_json(
            f"{manager_url.rstrip('/')}/api/dlg/original",
            {
                "experiment_id": payload["global_model"]["experiment_id"],
                "round": round_id,
                "client_id": client_id,
                "image": original_pixels(sample),
                "label": label,
            },
            timeout=30.0,
        )
        state.dlg_gradient = {"format": gradient["format"], "noise_sigma": sigma}
        state.dlg_round = round_id
        server_url = payload.get("server_url", current_app.config["SERVER_URL"])
        response = post_json(
            f"{server_url.rstrip('/')}/api/dlg/gradient",
            {
                "round": round_id,
                "client_id": client_id,
                "model_version": payload["global_model"].get("model_version", 0),
                "noise_sigma": sigma,
                "defense_enabled": bool(payload.get("defense_enabled", False)),
                "image_shape": [3, 32, 32],
                "gradient": gradient,
            },
            timeout=float(payload.get("round_timeout", 900)) + 120.0,
        )
        return jsonify(status="submitted", round=round_id, client_id=client_id,
                       server_status=response.get("status"))
'''
changes['clients/routes/experiment.py'] = [(
'''    if payload.get("global_model", {}).get("architecture") == MODEL_FORMAT:
        return jsonify(error="The role-2 DLG implementation must be connected to common.model before using the real CNN"), 501
''',
client_real
)]

# 2. 서버: 실제 그래디언트 복원. 구형 mock 분기와 인터페이스는 유지.
server_real = '''    if gradient.get("format") == "small_cnn_gradient_v1":
        from security.dlg_api import reconstruct_gradient_payload
        global_model = current_app.extensions.get("global_model")
        if not global_model:
            return jsonify(error="Global CNN model is missing"), 409
        try:
            result = reconstruct_gradient_payload(
                global_model, gradient,
                steps=int(config.get("dlg_iterations", 2)),
                seed=int(config.get("seed", 42)) + round_id,
            )
        except (ValueError, TypeError, RuntimeError, MemoryError) as exc:
            return jsonify(error=f"Real DLG reconstruction failed: {exc}"), 400
        dlg_result = {
            "round": round_id,
            "client_id": payload.get("client_id"),
            "model_version": payload.get("model_version"),
            "noise_sigma": payload.get("noise_sigma", 0.0),
            "defense_enabled": payload.get("defense_enabled", False),
            **result,
        }
        current_app.extensions.setdefault("dlg_results", {})[round_id] = dlg_result
        callback_url = config.get("experiment_manager_url")
        if not callback_url:
            return jsonify(error="DLG manager callback URL missing"), 409
        # 평가 관리자가 원본 이미지와 MSE를 계산하며, 서버는 원본을 알 수 없다.
        _post_json(callback_url.rstrip("/") + "/api/dlg/results", dlg_result)
        return jsonify(status="completed", round=round_id,
                       gradient_loss=dlg_result["gradient_loss"],
                       predicted_label=dlg_result["predicted_label"])
'''
changes['server/routes/dlg.py'] = [
('def _post_json(url, payload, timeout=3.0):', 'def _post_json(url, payload, timeout=180.0):'),
('''    config = current_app.extensions.setdefault("round_config", {})
    global_model = current_app.extensions.get("global_model") or {''',
'''    config = current_app.extensions.setdefault("round_config", {})
'''+server_real+'''    global_model = current_app.extensions.get("global_model") or {'''),
]

# 3. 관리자: 원본은 별도 평가 채널을 통해서만 수신, 결과 callback 종료 처리.
original_endpoint = '''@experiment_bp.post("/api/dlg/original")
def receive_dlg_original():
    """신뢰된 평가 관리자만 단일 이미지 원본을 보관한다. 서버 복원에는 전달하지 않는다."""
    payload = request.get_json(silent=True) or {}
    manager = current_app.extensions["experiment_manager"]
    try:
        manager.store_dlg_original(
            round_id=int(payload["round"]),
            experiment_id=payload["experiment_id"],
            client_id=int(payload["client_id"]),
            image=payload["image"],
        )
    except (KeyError, TypeError, ValueError) as exc:
        return jsonify(error=str(exc)), 400
    return jsonify(status="original_saved", round=payload["round"])


'''
changes['exp_manager/routes/experiment.py'] = [
('@experiment_bp.post("/api/dlg/results")\ndef receive_dlg_result():',
original_endpoint+'@experiment_bp.post("/api/dlg/results")\ndef receive_dlg_result():'),
]

# Remove obsolete DLG blockers, keep federated config rules unchanged.
old_validation = '''        if self.config.get("experiment_type") == "dlg" or any(item.get("round_type") == "dlg" for item in self.config.get("round_configs", [])):
            raise ValueError("실제 CNN의 DLG는 2번 담당자의 common.model 연결 후 사용할 수 있습니다.")
'''
old_round_block = '''        if round_type == "dlg":
            raise ValueError("실제 CNN의 DLG는 2번 담당자의 common.model 연결이 필요합니다.")
'''

# Original DLG simulator routes go through the server for model, client for gradients.
new_execute_dlg = '''    def execute_dlg_round(self, round_id: int, round_config: dict) -> dict:
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
        self.rest_client.post_json(
            f"{self.process_manager.server_url()}/api/round/config",
            round_config,
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

'''

new_record_dlg = '''    def record_dlg_result(self, round_id: int, result: dict) -> Path:
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

'''

# Long blocks are replaced based on method names; narrow enough to preserve all other methods.
def replace_between(text: str, start: str, end: str, replacement: str, path: str) -> str:
    a = text.find(start)
    b = text.find(end, a+len(start)) if a >= 0 else -1
    if a == -1 or b == -1 or text.count(start) != 1 or text.count(end) != 1:
        raise RuntimeError(f'{path}: method boundary mismatch: {start} to {end}')
    return text[:a] + replacement + text[b:]


def transform_runner(src: str) -> str:
    if 'def store_dlg_original(' in src and 'DLG 복원 및 MSE 평가가 완료' in src:
        return src
    src = apply_replacements(src, [
        (old_validation, ''),
        (old_round_block, ''),
    ], 'exp_manager/services/experiment_runner.py')
    src = replace_between(src,
        '    def execute_dlg_round(self, round_id: int, round_config: dict) -> dict:\n',
        '    def get_dlg_original(self, round_id: int) -> list[float]:\n',
        new_execute_dlg, 'exp_manager/services/experiment_runner.py')
    src = replace_between(src,
        '    def record_dlg_result(self, round_id: int, result: dict) -> Path:\n',
        '    def prepare_malicious_clients(\n',
        new_record_dlg, 'exp_manager/services/experiment_runner.py')
    return src


changes['exp_manager/services/experiment_runner.py'] = []

# 4. UI: show DLG iterations and keep them during settings save.
changes['exp_manager/templates/manager.html'] = [
('''<input id="dlg-noise" type="number" min="0" step="0.01" value="0" />''',
'''<input id="dlg-noise" type="number" min="0" step="any" value="0" />'''),
('''<label>사전 정상 연합학습 라운드 수<input id="dlg-training-rounds" type="number" min="1" value="5" /></label>''',
'''<label>사전 정상 연합학습 라운드 수<input id="dlg-training-rounds" type="number" min="1" value="5" /></label>
            <label>DLG 복원 반복 횟수<input id="dlg-iterations" type="number" min="1" max="1000" value="2" /></label>'''),
('''            dlg_training_rounds: Number(document.getElementById('dlg-training-rounds').value),''',
'''            dlg_training_rounds: Number(document.getElementById('dlg-training-rounds').value),
            dlg_iterations: Number(document.getElementById('dlg-iterations').value),'''),
('''document.getElementById('dlg-training-rounds').value = config.dlg_training_rounds ?? 5;''',
'''document.getElementById('dlg-training-rounds').value = config.dlg_training_rounds ?? 5;
        document.getElementById('dlg-iterations').value = config.dlg_iterations ?? 2;'''),
]


def main():
    branch = subprocess.check_output(
        ['git', 'branch', '--show-current'], cwd=ROOT, text=True
    ).strip()
    if branch != 'feature/security-integration':
        raise RuntimeError(
            f'현재 브랜치 {branch!r}에서 변경하지 않습니다. feature/security-integration으로 이동해 주세요.'
        )
    outputs = {}
    for path, replaces in changes.items():
        src_path = ROOT / path
        if not src_path.is_file():
            raise FileNotFoundError(f'코드가 없습니다: {src_path}')
        original = src_path.read_text(encoding='utf-8')
        changed = transform_runner(original) if path == 'exp_manager/services/experiment_runner.py' \
            else apply_replacements(original, replaces, path)
        if changed != original:
            outputs[path] = (original, changed)
    # 모든 검증 성공 후에만 파일을 기록한다.
    if not outputs:
        print('이미 적용된 DLG 2단계입니다. 변경할 파일이 없습니다.')
        return
    for path, (old, new) in outputs.items():
        backup_path = BACKUP / path
        backup_path.parent.mkdir(parents=True, exist_ok=True)
        if not backup_path.exists():
            backup_path.write_text(old, encoding='utf-8')
        (ROOT / path).write_text(new, encoding='utf-8')
        print('적용:', path)
    print('백업 폴더:', BACKUP)
    print('다음: python -m pytest -q')


if __name__ == '__main__':
    main()
