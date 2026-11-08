# DLG 실험 기록 (WORK_LOG.md에 복사하거나 별도로 보관)

## 2026-11-08 — DLG 기본 복원 기능 실습

### 실험 목적
- 이미지 한 장에서 계산한 가중치 그래디언트만으로 이미지를 복원할 수 있는지 확인
- 원본 이미지와 정답 레이블을 공격의 입력으로 사용하지 않고, 사후 평가에서만 사용

### 환경
- OS: Windows (사용자 PC)
- Python: 3.13.3 (앞서 확인한 버전)
- PyTorch: `python -c "import torch; print(torch.__version__)"` 출력 기록
- 입력: synthetic 8×8 RGB / CIFAR-10 train 이미지 32×32 RGB (실험별로 구분)
- 모델: TinyDLGNet (교육용 1계층 선형 분류기)
- 시드: 42
- optimizer: Adam, 학습률 0.1
- 반복 횟수: synthetic 60 / CIFAR-10 실험 시 명령어 기록
- 그래디언트 노이즈: 0 (4번째 보안 담당 기능에서 실험 예정)

### 수행 명령어
```powershell
python -m pytest -v
python -m experiments.dlg_experiment --source synthetic --steps 60
```

### 결과 기록 (실행 후 직접 입력)
- 기존 테스트: 8개 통과 여부: _____
- 새 DLG 테스트: 5개 통과 여부: _____
- 실험 이미지 ID: _____
- 그래디언트 일치 손실: _____ → _____
- 복원 이미지 MSE: _____
- 참 레이블(평가용): _____
- 예측 레이블: _____
- 저장된 파일: `results/dlg/synthetic/summary.json`, `comparison.png`, `loss_history.csv`

### 결과 해석
- 그래디언트 일치 손실 감소는 공격 최적화의 진행을 의미한다.
- MSE는 원본과 복원 결과의 픽셀 차이다. 실제 이미지 복원 성공 여부는 이미지와 MSE를 함께 확인한다.
- synthetic 8×8 결과는 최종 CIFAR-10 결과가 아니며, 모델 성능을 뜻하지 않는다.

### 다음 작업
- CIFAR-10 32×32 이미지 한 장으로 동일한 방식 실험
- 노이즈 없음/작음/큼 조건으로 DLG 결과와 MSE 비교
- 동일 모델·이미지·초기화·반복 횟수를 유지해 조건별 비교
- 팀의 실제 모델과 연동
