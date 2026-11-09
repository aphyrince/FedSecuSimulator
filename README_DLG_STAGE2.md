# FedSecuSimulator — DLG 2단계 Flask 연결

## 이 단계의 의미

1단계에서 성공한 `security/dlg.py` + 실제 `SmallCNN` 복원을 **Flask 클라이언트/서버/관리자에 연결**합니다.

- **클라이언트**: 자신에게 분배된 CIFAR-10의 첫 번째 이미지를 읽어 SmallCNN 그래디언트를 계산하고, 요청한 경우 가우시안 노이즈를 추가합니다.
- **중앙 서버**: 원본 이미지·정답 레이블을 전달받지 않고, 전역 모델과 공유 그래디언트만으로 DLG를 수행합니다.
- **평가 관리자**: 클라이언트로부터 별도 API로 받은 원본과 서버 복원 결과를 비교하여 MSE와 PNG를 저장합니다.
- **웹**: 기존 DLG 차단을 해제하고 복원 반복 횟수를 입력할 수 있게 합니다.

**중요:** 구현 후 기능을 개발 환경에서 통합 검증하는 단계입니다. 여기서 제공하는 코드는 사용자의 로컬 전체 저장소에서 아직 실행 검증하지 않았습니다. 새 핵심 함수 및 이미지 저장 기능에 대해서만 별도 3개 테스트가 통과했습니다. 꼭 모든 테스트와 웹 smoke test를 실행하세요.

## 적용 전 확인

- 현재 작업 브랜치 `feature/security-integration`.
- DLG 1단계 파일 `security/dlg.py`에는 `reconstruct_from_gradients(..., image_min=-1, image_max=1)` 지원이 있어야 합니다.
- 기존 `python -m pytest -q` 결과는 `60 passed, 15 subtests passed`.
- `results/` 는 로컬에 보관하고 GitHub에 대용량 모델을 올리지 마세요.

## 설치 (VS Code / PowerShell)

압축 파일 내부의 `security/`, `exp_manager/`, `tests/`, `tools/` 폴더를 FedSecuSimulator 프로젝트의 동일 폴더에 복사합니다. **기존 파일을 삭제하지 않습니다.** `dlg_api.py`, `dlg_report.py`, 새 테스트, 패치 스크립트만 새로 추가됩니다.

VS Code 프로젝트 최상위 터미널에서 다음 순서로 실행합니다.

```powershell
# 먼저 웹의 전체 프로세스 종료, 실험 관리자 터미널 Ctrl+C

git branch --show-current
git status --short

# 기존 기존 경로를 찾아 필요한 몇 군데만 수정하고 원본은 results/에 백업합니다.
python tools/apply_dlg_stage2.py

# 전체 회귀 테스트
python -m pytest -q

# 변경된 코드 목록
git status --short
```

문제가 생기면 **커밋하지 말고** 터미널의 전체 오류를 확인해 주세요. 백업은 `results/dlg_stage2_backup/`에 보관됩니다. 실험을 시작하기 전까지는 GitHub로 push할 필요가 없습니다.

## DLG 웹 시범 실행 (먼저 노이즈 OFF)

1. `python exp_manager/app.py` 실행, `http://127.0.0.1:8000` 접속.
2. **전체 프로세스 시작**. 중앙 서버와 클라이언트 10개 정상 확인.
3. **훈련/테스트 데이터 범위 분배** → `configured: True` 확인.
4. 실험 ID: `security_dlg_real_01` (이미 사용한 ID라면 새 ID).
5. 실험 유형: **DLG 실험 (정상 연합학습 후 공격)**.
6. **사전 정상 연합학습 라운드 수: 1**, **DLG 복원 반복 횟수: 2**.
7. 클라이언트: `0` (자동 선택해도 되지만 처음에는 0번 권장).
8. 정상 학습 설정: CPU, 참여율 1, 시드 42, epoch 1, 배치 32, 학습률 0.001, 클라이언트당 20장, 테스트 100장, 공격 OFF, 서버 방어 OFF.
9. **DLG 그래디언트 노이즈 방어 사용: OFF**.
10. **라운드 설정표 생성** → 1번(정상 연합학습)과 2번(DLG) 확인. 표의 공격/서버 방어는 모두 OFF.
11. **설정 저장**. 가능하면 API 설정도 점검:
    `Invoke-RestMethod http://127.0.0.1:8000/api/experiment/config | Select-Object experiment_id, experiment_type, dlg_training_rounds, dlg_iterations`
12. 파란색 **실험 시작** 클릭. 최종 `completed`를 확인.

기대 파일:

```text
results/security_dlg_real_01/
├── accuracy.csv 또는 accuracy.png
├── rounds.csv
├── results.json
├── dlg_result.json
├── dlg_original.png
├── dlg_reconstructed.png
└── dlg_comparison.png
```

여기서 원본 이미지는 클라이언트가 **평가 관리자**에게만 전달합니다. 중앙 서버의 `/api/dlg/participant`에는 원본을 보내지 않고, 서버의 `/api/dlg/gradient`에는 그래디언트만 전달합니다.

### 노이즈 실험

첫 번째 실험 완료 후 **상태 초기화**, 새 ID `security_dlg_noise_01`로 동일한 조건 설정. **DLG 그래디언트 노이즈 방어 ON**, `σ=0.001`, **설정 저장** 후 실험 실행. 실험 결과의 `mse`를 비교합니다.

초기 `dlg_iterations=2`는 연결 확인 용도이며 복원 품질을 평가하기에는 부족합니다. CPU/메모리 사용량을 확인하면서 반복 횟수를 늘리세요.

### 결과 확인 명령어

```powershell
$result = Get-Content .\results\security_dlg_real_01\dlg_result.json -Raw | ConvertFrom-Json
$result | Select-Object round, model_version, client_id, noise_sigma, defense_enabled, mse, gradient_loss, iterations
```

## 기존 코드와의 호환성

- `security/defenses.py`, `security/gradient_noise.py`, `security/dlg.py`는 덮어쓰지 않습니다.
- `clients/routes/experiment.py`, `server/routes/dlg.py`, `exp_manager/routes/experiment.py`, `exp_manager/services/experiment_runner.py`, `exp_manager/templates/manager.html`만 기존 특정 코드 위치를 찾아 수정합니다.
- 수정 전 각 파일은 `results/dlg_stage2_backup/`에 저장됩니다.
- 기존 가상 DLG 서비스의 코드 경로는 남겨두되, 실제 CNN(`small_cnn_v1`)인 경우만 새로운 계산 경로로 분기합니다.
- 결과 폴더는 `.gitignore`의 `/results/` 설정으로 기본 제외합니다.
- 로컬 실험 완료 후 `git diff`를 검토하고 3번 담당자에게 API 변경사항을 공유한 뒤 PR을 만드세요.

## 주의

- 이 구현은 교육용 로컬 신뢰 환경에서만 사용합니다. HTTP 인증·접근 통제가 구현되어 있지 않으므로 외부망에 서버를 노출하지 마세요.
- 가우시안 노이즈만으로 차등 개인정보보호(DP)가 보장되지는 않습니다.
- DLG의 MSE 증감은 이미지/시드/반복 횟수에 따라 달라집니다.
