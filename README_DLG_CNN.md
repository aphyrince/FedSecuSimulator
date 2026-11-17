# FedSecuSimulator DLG 실제 CNN 연결 — 1단계

이 ZIP의 `security/dlg.py`는 프로젝트 기존 파일의 단일 변경 버전입니다. 원래 0~1 범위 DLG 기능과 테스트를 유지하면서 `image_min`, `image_max` 선택 인자만 추가했습니다.

`experiments/dlg_cnn_experiment.py`는 실제 `fedsec.model.SmallCNN`과 `fedsec.data.CifarRangeDataset`을 사용합니다. **아직 Flask 서버 API를 수정하지는 않습니다.**

## 설치

1. VS Code에서 `feature/security-integration` 브랜치를 확인합니다.
2. ZIP 안의 `security/dlg.py`를 프로젝트 `security/dlg.py`에 덮어씁니다.
3. `experiments/dlg_cnn_experiment.py`를 프로젝트 `experiments/`에 복사합니다.
4. `tests/test_dlg_cnn.py`를 프로젝트 `tests/`에 복사합니다.
5. 프로젝트 최상위 디렉터리, 활성화된 `.venv`에서 명령을 실행합니다.

## 실행

```powershell
python -m pytest -q tests/test_dlg.py tests/test_gradient_noise.py tests/test_dlg_cnn.py

# 새로 초기화한 실제 SmallCNN 모델 구조로 실행
python -m experiments.dlg_cnn_experiment --steps 2 --noise-stds 0

# 이전에 저장된 실제 학습 체크포인트를 사용하려면
python -m experiments.dlg_cnn_experiment --checkpoint results/security_baseline_01/checkpoints/global_0001.pt --steps 2 --noise-stds 0

# 노이즈 유무 비교 (연산량이 높으므로 단계적으로 반복 횟수를 늘림)
python -m experiments.dlg_cnn_experiment --checkpoint results/security_baseline_01/checkpoints/global_0001.pt --steps 5 --noise-stds 0 0.001
```

기본 데이터 위치: `data/cifar-10-batches-py/test_batch`. 다른 위치면 `--batch-file`로 지정합니다.

출력: `results/dlg_cnn_stage1/`의 `original.png`, `reconstructed_std_*.png`, `comparison.png`, `loss_history_std_*.csv`, `summary.csv`, `summary.json`.

## 주의사항

- `SmallCNN`은 CIFAR-10을 -1~1 정규화된 입력으로 사용하므로 `image_min=-1`, `image_max=1` 범위로 복원합니다.
- 첫 2~5회는 복원 품질 평가가 아니라 경로와 2차 미분 실행 확인입니다. 품질 비교는 충분한 최적화 반복으로 재실험합니다.
- MSE는 원본과 복원 이미지를 0~1 픽셀 범위로 변환한 후 산출합니다.
- 단순 가우시안 노이즈는 차등 개인정보보호(DP)를 보장하지 않습니다.
- 결과는 `.gitignore`의 `/results/`에 따라 로컬에 남깁니다. 완성 코드만 선별 커밋하세요.
- 독립 실행 성공 후 클라이언트/서버 REST API를 연결합니다. 웹에서 DLG를 켜는 것은 아직 시기상조입니다.
