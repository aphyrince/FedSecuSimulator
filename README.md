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

## DLG 실험 자동 실행

웹 UI에서 실험 유형을 `DLG 실험 (정상 연합학습 후 공격)`으로 선택하고
`DLG 사전 정상 연합학습 라운드 수`를 설정합니다.
`POST /api/experiment/start`가 호출되면 실험 관리자는 설정된 횟수만큼 정상 연합학습
라운드를 순차 실행하고, 그 다음 라운드에서 자동으로 DLG 공격을 실행합니다.

## DLG 원본 이미지 경로 전달

실제 데이터 경로는 `exp_manager/.env`에 저장합니다.

```dotenv
DLG_IMAGE_PATH=data/dlg/sample.jpg
DLG_LABEL_PATH=data/dlg/sample_label.txt
```

실험 관리자는 이미지와 레이블 내용을 전달하지 않고 위 경로만 중앙 서버와
선정 클라이언트에 전달합니다. 클라이언트의 [clients/services/dlg_data_loader.py](clients/services/dlg_data_loader.py)가

## DLG 가우시안 노이즈 방어

웹 UI의 `DLG 그래디언트 노이즈 방어 사용`을 켜면 DLG 참가 클라이언트가 원본 이미지
그래디언트 계산 후 설정된 `DLG 가우시안 노이즈 σ`를 적용합니다. 방어가 꺼져 있으면
노이즈 sigma와 관계없이 노이즈를 추가하지 않습니다. 노이즈 크기는 음수가 될 수 없으며,
실험 설정과 DLG 결과 기록에 방어 여부와 실제 적용된 sigma가 포함됩니다. 클라이언트의 [clients/services/dlg_data_loader.py](clients/services/dlg_data_loader.py)가
파일을 읽고, [clients/services/dlg_service.py](clients/services/dlg_service.py)의
`compute_single_image_gradient()`가 전역 모델과 이미지로 그래디언트를 계산합니다.
실제 CIFAR 모델 구현으로 교체할 때는 `compute_single_image_gradient()`의 시그니처를 유지한 채
함수 본문만 교체하면 됩니다.

## DLG 더미 이미지 생성과 복원 최적화

중앙 서버의 [server/services/dlg_attack.py](server/services/dlg_attack.py)에 DLG 교체 지점이 분리되어 있습니다.

- `create_dummy_image(global_model, image_shape, seed) -> list[float]`: DLG 라운드 시작 시 학습된 체크포인트와 이미지 shape으로 더미 이미지를 생성합니다.
- `reconstruct_image(global_model, target_gradient, image_shape, label, seed, iterations) -> dict`: 중앙 서버가 받은 그래디언트와 더미 이미지를 이용해 복원 이미지를 최적화합니다.

실제 PyTorch/DLG 구현으로 교체할 때 두 함수의 파라미터와 반환 형식을 유지하면 REST 흐름을 그대로 사용할 수 있습니다.
복원 결과는 중앙 서버의 `POST /api/dlg/gradient`가 실험 관리자 `POST /api/dlg/results`로 전달하고,
실험 관리자가 원본 이미지와 복원 이미지의 MSE를 계산한 뒤 라운드 JSON에 기록합니다.

DLG 라운드의 참가 클라이언트는 실험 관리자가 자동으로 한 개 선정하거나,
웹 UI의 `DLG 참가 클라이언트`에서 ID를 지정할 수 있습니다.
선정 정보는 `POST /api/dlg/participant`를 통해 중앙 서버에 전달되고,
중앙 서버는 저장된 학습 완료 전역 모델 체크포인트를 해당 클라이언트의
`POST /api/dlg/start`에 전달합니다.

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

## 학습 전후 업데이트 추출

시나리오 11의 전역 모델 업데이트 추출은
[clients/services/local_training.py](clients/services/local_training.py)의 `extract_model_update()`가 담당합니다.

```python
def extract_model_update(
    global_model_before_training: dict,
    local_model_after_training: dict,
) -> dict:
    ...
```

함수는 학습 전후 모델에서 변경된 값을 추출하고 다음 형태로 반환해야 합니다.

```python
{
    "values": {},       # 파라미터별 학습 전후 차이
    "update_norm": 0.0, # 추출된 업데이트의 L2 norm
}
```

현재 기본 구현은 JSON 모델의 `weights`를 숫자 리스트·딕셔너리로 계산합니다. Tensor/PyTorch 모델로 교체할 때는
이 함수만 Tensor 차이를 계산하고 JSON 직렬화 가능한 형태 또는 Base64 payload로 변환하도록 변경하면 됩니다.

- `GET /api/updates/extracted/<round_id>`: 라운드별 추출 업데이트 조회
- 추출 업데이트는 로컬 학습 기록과 함께 저장됩니다.
- `POST /api/round/execute`의 `update.values` 및 `update.update_norm`에 반영됩니다.

## 클라이언트 업데이트 제출

시나리오 12에서는 추출된 로컬 업데이트가 클라이언트에서 중앙 서버로 직접 제출됩니다.

- 클라이언트: `POST /api/round/submit`
- 중앙 서버: `POST /api/rounds/<round_id>/updates`
- 중앙 서버 조회: `GET /api/rounds/<round_id>/updates`
- 클라이언트 제출 확인: `GET /api/updates/submitted/<round_id>`

예시:

```powershell
Invoke-RestMethod -Method Post `
  -Uri http://127.0.0.1:5101/api/round/submit `
  -ContentType application/json `
  -Body '{"round":1,"malicious":false}'
```

## 업데이트 수신 및 유효성 검사

중앙 서버는 제출된 업데이트가 아래 조건을 만족하는지 확인합니다.

- 제출 라운드가 요청한 라운드와 일치
- 클라이언트가 해당 라운드에 선정됨
- `valid`가 `true`
- `num_samples`가 음수가 아닌 숫자
- `update` 객체가 존재

API 조회:

- `GET /api/rounds/<round_id>/updates/status`: 제출·미제출·기한 초과 클라이언트 확인
- `POST /api/rounds/<round_id>/timeout`: 기한 초과 처리 및 미제출 클라이언트 알림

라운드 설정의 `round_timeout` 또는 `update_timeout`(초)로 제출 기한을 설정할 수 있습니다.

## FedAvg 전역 모델 갱신

시나리오 14의 FedAvg 집계는 [server/services/fedavg.py](server/services/fedavg.py)의
`aggregate_fedavg()`가 담당합니다.

```python
def aggregate_fedavg(
    global_model: dict,
    client_updates: list[dict],
) -> dict:
    ...
```

각 클라이언트 업데이트에는 다음 정보가 필요합니다.

```python
{
    "client_id": 0,
    "num_samples": 100,
    "valid": True,
    "local_model": {"model_version": 1, "weights": {}},
}
```

기본 구현은 `num_samples`를 표본 가중치로 사용해 JSON `weights`의 숫자 값을 평균합니다. PyTorch 모델로 교체할 때:

1. 이 함수 안에서 Tensor 로딩과 가중 평균을 구현합니다.
2. 반환값의 `global_model`, `total_samples`, `client_weights` 키는 유지합니다.
3. Tensor를 API로 전달할 때는 JSON 직렬화 함수로 변환하고 다음 라운드에서 복원합니다.

갱신 API와 조회 API는 다음과 같습니다.

- `GET /api/model/global`: 현재 FedAvg 전역 모델
- `GET /api/model/history`: 라운드별 갱신된 전역 모델과 표본 가중치

## 전역 모델 성능 평가

시나리오 15의 테스트 데이터셋 평가는
[server/services/model_evaluation.py](server/services/model_evaluation.py)의
`evaluate_global_model()`이 담당합니다.

```python
def evaluate_global_model(
    global_model: dict,
    dataset_range: dict,
    evaluation_config: dict,
    round_id: int,
) -> dict:
    ...
```

반환값 필수 키:

```python
{
    "accuracy": 0.0,
    "loss": 0.0,
    "num_test_samples": 0,
    "correct_predictions": 0,
    "metrics": {},
}
```

실제 CIFAR-10 평가로 교체할 때는 함수 본문에서 모델 가중치 복원, 테스트 범위의
이미지·레이블 로딩, `model.eval()` 추론을 구현하고 위 형식으로 반환하면 됩니다.

- `GET /api/evaluation/<round_id>`: 해당 라운드 평가 결과 조회
- `GET /api/evaluation/history`: 전체 라운드 평가 결과 조회

평가 결과는 중앙 서버의 `evaluation_results` 상태와 라운드 결과에 함께 저장됩니다.

## 라운드 결과 전달

중앙 서버는 라운드 집계 완료 후 실험 관리자에게 다음 API로 결과를 전달합니다.

- 중앙 서버: `POST /api/rounds/<round_id>/start`
- 실험 관리자 수신: `POST /api/results/round`

전달 payload에는 다음 데이터가 포함됩니다.

- 모델 평가 결과: `accuracy`, `loss`, `evaluation`
- 갱신된 전역 모델: `global_model`
- 라운드에 제출된 클라이언트 업데이트: `client_updates`
- 선정·채택·지연·악성 클라이언트 정보
- FedAvg 표본 가중치

실험 관리자 설정의 `experiment_manager_url`을 중앙 서버로 전달하면 자동 콜백됩니다.
실험 관리자 프로세스가 설정한 호스트와 포트를 사용해 자동 설정되며, 기본값은 `http://127.0.0.1:8000`입니다.

## 라운드 결과 파일 기록

실험 관리자는 라운드 결과 수신 즉시 다음 위치에 JSON 파일을 기록합니다.

`results/<experiment_id>/rounds/round_<번호>.json`

전체 실험 실행 중 다음 라운드가 예약되어 있으면 수신 API가 중앙 서버의
`POST /api/rounds/<round_id>/start` 호출을 순차적으로 시작합니다. 마지막 라운드가
완료되면 기존 `results.json`, `rounds.csv`, 정확도 그래프도 생성됩니다.

## 내부자 협력 공격 라운드

라운드 설정에서 다음 값을 사용합니다.

- `attack_enabled: true`
- `malicious_clients: [클라이언트 ID 목록]`
- `attack_scale: -5.0`
- 방어: `defense`를 `none`, `norm_clipping`, `coordinate_median` 중 하나로 설정
- 노름 제한: `clip_norm`

악성 클라이언트는 로컬 업데이트를 생성한 뒤 `/api/round/collusion`으로 서로의
업데이트를 공유합니다. 공동 평균에 `attack_scale`을 곱한 변조 업데이트를 중앙 서버에
제출합니다. 공격 교환 구현은 `clients/routes/experiment.py`의
`collude_round_updates()`를 교체할 수 있습니다.

악성 클라이언트 추출값 교환은 `POST /api/round/update-share`를 사용합니다.
송신자는 본인의 추출 업데이트 payload를 상대 악성 클라이언트로 전달하고, 상대는
클라이언트 상태의 라운드별 `received_malicious_updates`에 저장합니다. 이후
`POST /api/round/collusion`은 자신의 추출값과 수신한 값의 평균에 공격 배율을 적용합니다.

## DLG 공격 교체 지점

DLG 라운드 설정은 `round_type=dlg`, `dlg_noise_sigma`, `dlg_iterations`,
`dlg_image`, `dlg_image_shape`, `dlg_label`로 구성합니다.

- 클라이언트 그래디언트 계산: `clients/services/dlg_service.py`
  - `compute_single_image_gradient(global_model, image, label)`
  - `add_gaussian_noise(gradient, sigma, seed)`
- 서버 DLG 복원: `server/services/dlg_attack.py::reconstruct_image(...)`
- 클라이언트 API: `POST /api/dlg/start`
- 서버 API: `POST /api/dlg/gradient`
- 관리자 MSE 수신: `POST /api/dlg/results`

팀 권장 가우시안 노이즈 실험값은 `0`, `0.01`, `0.05`, `0.10`입니다.
실제 PyTorch/CNN 구현으로 교체할 때도 위 함수 시그니처와 JSON
`gradient.values`, `reconstructed_image` 반환 형식을 유지하면 REST 흐름은 그대로 사용할 수 있습니다.

## 방어 알고리즘 교체

라운드 설정에는 `defense_enabled`, `defense`, `clip_norm`이 전달됩니다.

- `defense_enabled=false`: 방어 로직을 실행하지 않고 일반 FedAvg를 사용합니다.
- `defense_enabled=true`, `defense=norm_clipping`: [byzantine_defense.py]의
  `apply_norm_clipping(updates, clip_norm)`을 실행한 뒤
  [fedavg.py]의 `aggregate_fedavg()`로 집계합니다.
- `defense_enabled=true`, `defense=coordinate_median`: [fedavg.py]의
  `aggregate_coordinate_median()`으로 좌표별 중앙값 집계를 수행합니다.

실제 PyTorch 텐서 구현으로 교체하려면 반환 형식은 유지한 채 다음 함수만 구현하면 됩니다.

- 노름 클리핑: `server/services/byzantine_defense.py::apply_norm_clipping`
- 좌표별 중앙값 집계: `server/services/fedavg.py::aggregate_coordinate_median`

클라이언트 update payload의 `update`, `local_model`, `num_samples` 형태는 그대로 유지해야 합니다.
