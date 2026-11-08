# 팀 구현 로직 교체 가이드

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
    "model_delta": 0.0,
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
