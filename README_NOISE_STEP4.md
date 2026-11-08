# FedSecuSimulator 2번 담당: 4번째 기능 — DLG 가우시안 노이즈 방어

기존 `security/dlg.py`, `security/defenses.py`, `experiments/dlg_experiment.py`를 수정하지 않고 **새 파일만 추가**하는 방식입니다.

## 1. 파일 복사
프로젝트 루트 `FedSecuSimulator/`에 압축 안의 폴더 구조를 그대로 복사하세요. 동일 파일이 있다면 내용을 확인 후 병합합니다.

- `security/gradient_noise.py`: 그래디언트에 `N(0, std^2)` 추가
- `experiments/dlg_noise_experiment.py`: 동일 조건에서 노이즈 0 / 0.01 / 0.1 비교
- `tests/test_gradient_noise.py`: 결과·재현성·예외·기존 기준선 검증
- `docs/DLG_NOISE_LOG.md`: 보고서용 로그 양식

## 2. Windows PowerShell 실행
프로젝트 최상위에서 가상환경 Python을 사용하세요.

```powershell
python -m pytest -v
python -m experiments.dlg_noise_experiment --source synthetic --steps 60
```

합성 이미지 결과: `results/dlg_noise/synthetic/summary.csv`, `summary.json`, `comparison.png`, 조건별 복원 PNG 및 손실 CSV.

CIFAR-10 실험(이미 `torchvision`이 설치된 경우):

```powershell
python -m experiments.dlg_noise_experiment --source cifar10 --index 25 --steps 200 --out-dir results/dlg_noise/cifar10_25
```

최초 CIFAR-10 다운로드는 인터넷이 필요하고 CPU는 오래 걸릴 수 있습니다.

## 3. 데이터 사용·해석 주의
- 원본 이미지와 레이블은 **그래디언트 생성 + 복원 종료 후 평가**에만 사용합니다.
- 세 조건 모두 **같은 원본·모델·초기 복원 이미지·반복 횟수**를 사용합니다.
- MSE가 높다고 항상 더 좋은 보안이라고 일반화할 수는 없습니다. 다양한 시드·이미지에서 비교하세요.
- 단순 Gaussian noise 추가는 정식 **차등 개인정보보호(DP) 보장**과 다릅니다.
- 노이즈 수치 0/0.01/0.1은 *초기 실험값*이므로 실제 프로젝트 최종 비교에서는 근거와 함께 고정해야 합니다.

## 4. Git 업로드

```powershell
git branch --show-current
git status
git add security/gradient_noise.py experiments/dlg_noise_experiment.py tests/test_gradient_noise.py docs/DLG_NOISE_LOG.md README_NOISE_STEP4.md
git commit -m "feat: add Gaussian gradient noise defense for DLG"
git push -u origin HEAD
```

출력 CSV/PNG는 검토 후 팀의 결과 보관 원칙에 따라 선택적으로 업로드하세요.
