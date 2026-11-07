# FedSecuSimulator 작업 기록

> 작성자: 2번 담당 (보안 알고리즘) / 저장소: FedSecuSimulator

## 2026-11-07 — 초기 수치 예제로 공격·방어 검증

### 목표
- 클라이언트 10개의 가상 업데이트를 사용해 악성 협력 공격(-5배) 구현 흐름 검증
- 정상/공격 상황에서 FedAvg, 클리핑 후 FedAvg, 클리핑 후 중앙값을 비교
- 보고서에 활용할 CSV 로그 자동 생성

### 환경
- OS: Windows
- Python: 3.13.3 (사용자 환경)
- 실행 명령: `python -m experiments.attack_defense_demo`
- 테스트 명령: `python -m pytest tests/test_defenses.py tests/test_attack_defense_demo.py -v`
- PyTorch 버전: **실행 후 `python -c "import torch; print(torch.__version__)"`로 확인하여 기록**
- Git 커밋: **커밋 후 `git rev-parse --short HEAD`로 확인하여 기록**

### 실험 설정
| 항목 | 값 |
|---|---|
| 클라이언트 | 10개 (ID 0~9) |
| 지정 악성 클라이언트 | 8, 9 (총 2개) |
| 참여율 | 100% |
| 업데이트 | 각 클라이언트의 2차원 **모의** 가중치 변화량 |
| 표본 수 | 각 100개로 가정 (가중치 동일) |
| 공격 | 참여 악성 클라이언트의 정상 업데이트 평균 × (-5), 동일하게 제출 |
| 클리핑 상한 | 0.7 (**실제 학습 전에 재설정할 임시값**) |
| 방어 비교 | FedAvg / Clipping+FedAvg / Clipping+Median |
| 랜덤 시드 | 불필요 (입력 벡터가 고정되어 있음) |

### 결과 파일
- `results/attack_defense_demo/summary.csv` : 6개 실험 조건 집계 결과
- `results/attack_defense_demo/client_update_log.csv` : 클라이언트별 클리핑 전후 노름

### 관찰 결과
- 실행 후 여기서 `summary.csv`의 값을 확인하고 기록할 것.
- **주의:** 이번 단계에는 실제 CIFAR-10 학습, 모델 정확도, 복원 MSE가 없음.

### 해석과 한계
- 이번 단계는 함수 단위 동작 검증이다. 업데이트가 정상 FedAvg 결과에 가까워졌다고 해서 실제 이미지 분류 정확도가 개선됐다고 결론낼 수는 없다.
- 실제 실험에서는 정상 학습 업데이트의 노름 분포를 보고 클리핑 상한을 정한 뒤 동일 조건에서 고정해야 한다.
- 지정 악성 ID는 예제 공격 생성과 로그 작성에만 쓰고 방어 함수에는 전달하지 않는다.

### 다음 할 일
- 1번 담당의 실제 로컬 학습 업데이트 및 FedAvg 인터페이스 연결
- 3번 담당의 서버/라운드/로그 구조와 통합
- 정상 실험으로 클리핑 상한 결정, 다중 시드/정확도 실험 진행
- 이후 별도 DLG 및 가우시안 노이즈 방어 구현
