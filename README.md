# FedSecuSimulator

2026 웹 기반 시스템 팀 프로젝트 5팀 연합학습 보안 시뮬레이션 및 공격·방어 결과의 원인 분석 (A조)

## Flask 프로세스 실행

실험 관리자만 실행하면 중앙 서버 1개와 클라이언트 10개가 함께 실행됩니다.

```powershell
.\.venv\Scripts\python.exe exp_manager\app.py
```

실험 관리자 웹 UI: http://127.0.0.1:8000

- **전체 프로세스 시작**: 중앙 서버와 클라이언트 프로세스를 다시 시작합니다.
- **전체 프로세스 종료**: 자식 프로세스만 REST API로 종료합니다.
- **실험 관리자 종료**: 자식 프로세스를 종료한 뒤 실험 관리자도 종료합니다.
- **실험 실행**: 웹 UI에서 라운드 수, 참여율, 시드, 공격·방어·데이터 분포를 설정하고 가상 연합학습을 실행합니다.

실험 관리 API:

- `GET /api/experiment/config`: 실험 설정 조회
- `POST /api/experiment/config`: 실험 설정 변경
- `POST /api/experiment/start`: 실험 시작
- `POST /api/experiment/stop`: 실험 중단 요청
- `GET /api/experiment/status`: 라운드 진행 및 결과 조회
- `POST /api/experiment/reset`: 실험 상태 초기화

실험 결과는 `results/<experiment_id>/` 아래에 저장됩니다.

- `rounds.csv`: 라운드별 선정·채택·지연·악성 참여·정확도 로그
- `results.json`: 재실행 설정과 라운드 결과
- `accuracy.csv` 또는 `accuracy.png`: 라운드별 정확도 결과 (환경에 matplotlib가 없으면 CSV로 저장)

## CIFAR-10 데이터 분배

`exp_manager/.env`에서 CIFAR-10 데이터 위치를 설정합니다.

```dotenv
SOURCE_DIR=source
TRAIN_DATA_FILE=data_batch_1,data_batch_2,data_batch_3,data_batch_4,data_batch_5
TEST_DATA_FILE=test_batch
```

`TRAIN_DATA_FILE`과 `TEST_DATA_FILE`은 쉼표로 구분하거나, CIFAR 공식 배치 파일 이름을 그대로 사용할 수 있습니다. 파일을 지정하지 않으면 `SOURCE_DIR` 아래의 표준 `cifar-10-batches-py` 폴더 또는 루트의 `data_batch_*`, `test_batch`를 자동 탐색합니다.

실험 관리자 실행 후 다음 API로 훈련 데이터 범위를 10개 클라이언트에 균등 분배하고, 테스트 범위를 중앙 서버에 전달합니다.

- `GET /api/data/status`: `.env` 설정과 분배 상태 조회
- `POST /api/data/distribute`: 데이터 범위 계산 및 분배
- `POST /api/dataset/range`: 중앙 서버와 클라이언트에서 받은 데이터 범위 조회용 개별 API

분배 요청 예시:

```powershell
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/api/data/distribute -ContentType application/json -Body '{"seed":42}'
```

실험 관리자가 종료되면 다음 REST API가 호출됩니다.

- 중앙 서버: `POST http://127.0.0.1:5000/api/shutdown`
- 클라이언트 0~9: `POST http://127.0.0.1:5101/api/shutdown`부터 순서대로 호출

개별 앱은 다음 명령으로도 실행할 수 있습니다.

```powershell
.\.venv\Scripts\python.exe server\app.py
.\.venv\Scripts\python.exe clients\app.py --client-id 0 --port 5101
```

## 전역 모델 구조 변경

실험 시작 시 중앙 서버의 랜덤 전역 모델은 [server/config/model_state.py](server/config/model_state.py)의
`create_random_global_model()`에서 생성됩니다.

팀원이 실제 PyTorch 모델을 연동할 경우:

1. 해당 함수의 반환 형태를 팀 모델 직렬화 형태로 변경합니다.
2. 중앙 서버 전달 payload는 기본적으로 Flask가 JSON으로 직렬화할 수 있어야 합니다.
3. Tensor 모델의 경우 Tensor를 중첩 리스트 또는 Base64로 변환한 후 각 클라이언트에서 원래 타입으로 복원합니다.
4. Flask JSON으로 직접 보낼 수 있는 간단한 모델 형태 예시는 다음과 같습니다.

```python
return {
    "model_version": 0,
    "architecture": "CNN",
    "num_parameters": 32,
    "weights": {
        "conv1.weight": [0.1, -0.2, 0.3],
        "conv1.bias": [0.0, 0.1],
    },
}
```

## 로컬 학습 로직 변경

전역 모델 수신 시 로컬 학습은 [clients/services/local_training.py](clients/services/local_training.py)의
`train_local_model()`이 담당합니다.

현재 포함된 함수는 아직 실제 학습을 수행하지 않는 가상 구현입니다. 팀원이 실제 CIFAR-10 학습을 구현할 때는
함수 이름과 아래 파라미터, 반환 형식을 유지하고 함수 본문만 교체하면 됩니다.

```python
def train_local_model(
    global_model: dict,
    dataset_range: dict,
    training_config: dict,
    round_id: int,
    client_id: int,
) -> dict:
    ...
```

반환값 필수 키:

```python
{
    "local_model": {},       # 학습 후 로컬 전역 모델 payload
    "model_delta": 0.0,      # 중앙 서버 집계가 사용할 업데이트 또는 업데이트 dict
    "num_samples": 0,        # 학습에 사용한 표본 수
    "training_loss": 0.0,    # 로컬 학습 손실
}
```

클라이언트 API 동작:

- `POST /api/model/global`: 전역 모델 교체 후 `train_local_model()`을 자동 실행합니다.
- `GET /api/training/status`: 최신 로컬 학습 상태와 결과를 조회합니다.
- `GET /api/training/history`: 라운드별 로컬 학습 기록을 조회합니다.
- `POST /api/round/execute`: 저장된 최신 로컬 학습 결과를 업데이트로 반환합니다.
