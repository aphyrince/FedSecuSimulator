# 팀 구현 로직 교체 가이드

## master 통합 상태

CNN·학습·FedAvg·실제 평가·협력 공격의 원본은 1번 담당 `fedsec/`에 있습니다.
공통 모델은 `fedsec/model.py`의 `SmallCNN`이며 RGB 32×32 입력, 10 logits, 평균/표준편차 0.5 전처리입니다.
`common.model.SmallCNN` 등의 기존 import 경로도 같은 구현을 가리킵니다.
정상·공격 비교는 `fedsec/comparison.py:compare_accuracy_results()`에서 구현하며 기존 `common.comparison`도 유지합니다.

### 담당 경계와 재사용 방법

- 1번: `fedsec`의 모델·학습·Tensor FedAvg·협력 공격·정확도 비교. 독립 실행은 `python -m fedsec --config configs/fedsec.example.json`.
- 2번: 방어·DLG 알고리즘. 기존 클리핑/중앙값 로직은 이번 분리에서 변경하지 않습니다.
- 3번: REST API·프로세스·분할 정책·참여율/지연·시스템 로그/시각화. 기존 경로로 1번 함수를 호출할 수 있습니다.

`fedsec.aggregate_fedavg(global_weights, updates)`의 입력은 Tensor 상태와
`ClientUpdate(client_id, num_samples, delta)` 목록이고 반환값은 새 Tensor 상태입니다.
서버의 `server.services.fedavg.aggregate_fedavg()`는 기존 요청 검증/JSON 변환 후 이 함수를 호출합니다.
`fedsec.train_local_model(model, dataset, settings, ...)`은 `LocalTrainingResult`를,
`fedsec.evaluate_global_model(model, dataset, settings, ...)`은 `EvaluationResult`를 반환합니다.
로컬 학습과 평가의 기존 서비스 함수는 이 결과를 기존 API 딕셔너리로 변환합니다.

1번 담당 핵심 코드에서 Flask나 `server`/`clients`/`exp_manager`를 import하지 마세요.
학습 계산을 바꿀 때는 `fedsec`를 수정하고, 전송 키나 기한·실험 ID 검증은 어댑터에서 처리합니다.
독립 실행 데이터 배정표는 예제일 뿐, 팀원의 실제 분할을 대신하지 않습니다.
참여율·지연·방어 실험은 기존 매니저 경로를 사용합니다.

추가 업데이트 메타데이터는 `experiment_id`, `base_model_version`, `base_model_hash`, `dataset_fingerprint`입니다.
`update.values`는 **파라미터별 실제 delta(학습 후 − 시작)**이고, `update.model_delta`와 `update.update_norm`은 노름 요약값입니다.
서버는 delta와 `local_model.weights`의 일치도 검사합니다. 공격/클리핑을 수정할 때 둘을 함께 갱신해야 합니다.
이를 위한 공통 함수가 `common.updates.with_delta()`입니다. `num_samples`에 epoch 수를 곱하지 않습니다.
집계·방어 함수는 악성 ID/라벨을 판단에 사용하지 않습니다. 현재 REST 오케스트레이터의 공격 메타데이터 분리는 3번 담당자와 조정할 수 있습니다.

`common.model.save_model_checkpoint()`의 `.pt`는 같은 모델 설정·전처리·가중치를 저장합니다.
DLG 담당자는 `load_model_checkpoint()`와 `build_model()`을 재사용하세요. 기존 가상 DLG는 실제 CNN 경로에서 실행하지 않습니다.

프로세스 내부 import는 `clients.*`, `server.*`, `exp_manager.*`의 절대 패키지 이름을 사용합니다.
같은 Python 프로세스에서 테스트하거나 모듈로 가져올 때 `services`/`config` 이름이 충돌하는 문제를 방지합니다.

이 문서의 함수는 팀원이 실제 모델·학습·공격·방어 로직으로 교체할 위치입니다.
REST payload의 주요 키는 유지해야 기존 API와 웹 UI가 정상 동작합니다.

## 전역 모델

실험 시작 시 중앙 서버의 랜덤 전역 모델은
[server/config/model_state.py](server/config/model_state.py)의
`create_random_global_model()`에서 생성됩니다.

실제 PyTorch 모델로 교체할 때 Tensor는 JSON 전달 전에 중첩 리스트 또는 Base64로 변환하고,
클라이언트에서 원래 Tensor 형태로 복원합니다.

## 로컬 학습과 업데이트 추출

교체 대상:

- [clients/services/local_training.py](clients/services/local_training.py)의 `train_local_model()`
- [clients/services/local_training.py](clients/services/local_training.py)의 `extract_model_update()`

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
    "local_model": {},
    "model_delta": {"parameter_name": [0.0]},
    "num_samples": 0,
    "training_loss": 0.0,
}
```

`extract_model_update()` 반환값은 다음 형식을 유지합니다.

```python
{
    "values": {},
    "update_norm": 0.0,
}
```

## FedAvg 집계와 모델 평가

교체 대상:

- [server/services/fedavg.py](server/services/fedavg.py)의 `aggregate_fedavg()`
- [server/services/model_evaluation.py](server/services/model_evaluation.py)의 `evaluate_global_model()`

```python
def aggregate_fedavg(
    global_model: dict,
    client_updates: list[dict],
) -> dict:
    ...
```

클라이언트 업데이트 payload 필수 키:

```python
{
    "client_id": 0,
    "num_samples": 100,
    "valid": True,
    "local_model": {"model_version": 1, "weights": {}},
}
```

FedAvg 반환값은 `global_model`, `total_samples`, `client_weights`를 유지합니다.

평가 반환값 필수 키:

```python
{
    "accuracy": 0.0,
    "loss": 0.0,
    "num_test_samples": 0,
    "correct_predictions": 0,
    "metrics": {},
}
```

## 내부자 협력 공격

교체 대상은 [clients/routes/experiment.py](clients/routes/experiment.py)의
`collude_round_updates()`입니다. 교체 후 다음 payload 형식을 유지합니다.

```python
{
    "round": 1,
    "client_id": 0,
    "update": {"values": {}, "update_norm": 0.0},
    "num_samples": 100,
}
```

## 서버 방어 알고리즘

교체 대상:

- [server/services/byzantine_defense.py](server/services/byzantine_defense.py)의 `apply_norm_clipping()`
- [server/services/fedavg.py](server/services/fedavg.py)의 `aggregate_coordinate_median()`

라운드 설정은 다음 키를 유지합니다.

- `defense_enabled`
- `defense`
- `clip_norm`

`defense=norm_clipping`이면 노름 클리핑 후 FedAvg를 실행합니다.
`defense=coordinate_median`이면 중앙값 집계 결과를 전역 모델로 사용합니다.
클라이언트 update payload의 `update`, `local_model`, `num_samples` 형태는 유지합니다.

## DLG 그래디언트 계산

교체 대상은 [clients/services/dlg_service.py](clients/services/dlg_service.py)입니다.

```python
def compute_single_image_gradient(
    model: dict,
    image: list[float],
    label: int,
) -> dict:
    ...

def add_gaussian_noise(
    gradient: dict,
    sigma: float,
    seed: int,
) -> dict:
    ...
```

`compute_single_image_gradient()` 반환값의 `gradient.values` 형식과
`add_gaussian_noise()`의 `gradient.values` 형식을 유지합니다.

## DLG 더미 이미지 생성과 복원

교체 대상은 [server/services/dlg_attack.py](server/services/dlg_attack.py)입니다.

```python
def create_dummy_image(
    global_model: dict,
    image_shape: list[int],
    seed: int,
) -> list[float]:
    ...

def reconstruct_image(
    global_model: dict,
    target_gradient: dict,
    image_shape: list[int],
    label: int,
    seed: int,
    iterations: int = 200,
    dummy_image: list[float] | None = None,
) -> dict:
    ...
```

`reconstruct_image()` 반환값에는 최소한 다음 키가 필요합니다.

```python
{
    "reconstructed_image": [],
    "image_shape": [],
    "gradient_loss": 0.0,
    "iterations": 200,
}
```

DLG REST payload는 다음 키를 유지합니다.

- 클라이언트 → 서버: `round`, `client_id`, `gradient.values`, `image_shape`, `label`, `noise_sigma`
- 서버 → 실험 관리자: `reconstructed_image`, `image_shape`, `gradient_loss`, `iterations`

팀 권장 가우시안 노이즈 실험값은 `0`, `0.01`, `0.05`, `0.10`입니다.
