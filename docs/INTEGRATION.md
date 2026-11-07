# 팀원 연결 규칙

제시된 API/역할/시나리오를 알고리즘과 연결하는 문서입니다.
HTTP 엔드포인트가 이 저장소에 구현되어 있다는 뜻은 아닙니다.

## 3번 · API 담당자

| API | 연결 |
| --- | --- |
| 매니저 → 클라이언트 `POST /dataset` | 파일을 Dataset으로 만들고 저장. 실제 길이와 class_distribution 합계 검증 |
| 서버 → 클라이언트 `POST /participation/check` | 데이터 준비·작업 상태를 확인해 참여 가능 여부 결정 |
| 클라이언트 → 서버 `POST /participation/response` | 실제 참여 응답. check HTTP 응답은 접수 확인으로 구분 권장 |
| 서버 → 클라이언트 `POST /train` | 접수 응답 후 작업에서 `train_from_checkpoint()` 호출 |
| 매니저 → 클라이언트 `POST /attack/configure` | `AttackConfig`로 저장. 정상 클라이언트에는 enabled=false |
| 협력 대상 전달 `POST /attack/peers` | 실제 참여 악성 ID/주소. 악성 ID를 서버 방어에 숨기려면 매니저가 전달 |
| 클라이언트 간 `POST /attack/updates` | 변조 전 업데이트를 수신해 라운드별 `PeerUpdateBuffer.add()` |
| 클라이언트 간 `GET /attack/status/{round_id}` | `PeerUpdateBuffer.status()` 결과 |
| 클라이언트 → 서버 `POST /updates` | `decode_update()` 후 배정 기록과 함께 검증 |
| 서버 집계 | `fedavg()` 또는 2번의 방어/집계 함수 |
| 서버 → 매니저 `POST /results/rounds` | `result.json`과 같은 구조로 전달 |

`model_url` 상대경로는 서버 기준 URL로 해석하세요. URL 예시의 `< >`는 실제 주소에서 제거합니다.
주소/포트/참여 기한/제출 기한은 3번 네트워크 설정에서 관리하세요.
독립 실행기는 네트워크 기한이나 실제 프로세스를 흉내 내지 않습니다.

### 클라이언트 정상 학습

```python
from fedsec.artifacts import encode_update
from fedsec.integration import train_from_checkpoint

# HTTP handler가 model_url 파일을 받은 뒤 백그라운드 작업에서 실행.
result = train_from_checkpoint(
    checkpoint_path="received/global.pt",
    payload=train_json,
    dataset=assigned_dataset,  # 정규화된 [3, 32, 32] 이미지, 0~9 정수 레이블
    client_id=client_id,
    experiment_id=current_experiment_id,
    data_config=config.data,
    training_defaults=config.training,
    runtime=config.runtime,
    seed=config.seed,
)
metadata = result.update.metadata()
binary = encode_update(result.update)
# 정상: metadata와 binary를 multipart /updates로 제출.
# 악성: 아래 협력 교환 후 변조한 업데이트 제출.
```

`/train`은 명세의 `round_id`, `model_version`, `local_epochs`, `batch_size`, `learning_rate`,
`model_url` 이름을 사용합니다. JSON에 `experiment_id`가 없으면 현재 실험 문맥을 주입합니다.
JSON에도 있으면 문맥과 일치해야 합니다. `resolve_training_config()`가 기본 설정에
요청의 epoch·배치·학습률을 반영해 최종 `TrainingConfig` 하나를 만듭니다.
옵티마이저 종류·momentum·weight_decay는 기본 설정에서 유지합니다.
`train_local()`은 이 최종 설정을 `training_config`로, 라운드 정보는 `context`로 받습니다.
HTTP 요청 객체를 학습 함수에 직접 전달하지 않습니다. 전처리는 실험 공통 설정입니다.
매 라운드 옵티마이저를 초기화하는 규칙을 사용합니다.
호출마다 독립 모델을 만들지만 전역 난수 시드를 쓰므로 같은 프로세스의 학습 스레드를 동시에 실행하지 마세요.
클라이언트별 작업을 직렬화하거나 별도 프로세스에서 실행하세요.

### 클라이언트 협력 공격

```python
from fedsec.attack import PeerUpdateBuffer
from fedsec.artifacts import decode_update

inbox = PeerUpdateBuffer(result.update, expected_clients=participating_malicious_ids)
# result.update(변조 전 값)를 다른 협력자에게 전송.
# /attack/updates로 받은 바이너리마다:
inbox.add(decode_update(peer_binary))
# 전원 수신 또는 기한 도달 시:
outcome = inbox.finalize(attack_config)
own_submission = next(u for u in outcome.updates if u.client_id == client_id)
# own_submission만 서버 /updates에 제출.
```

수신함은 `(experiment_id, round_id, model_version)`별로 생성하고 완료 후 정리하세요.
예상 목록에 자기 자신을 포함하고 비참여자는 제외합니다.
중복 수신은 오류이므로 HTTP 재시도는 어댑터에서 동일 요청인지 확인해 처리하세요.
부분 수신 시 클라이언트별 목록이 달라질 수 있습니다. `available_only`를 사용하면
참여자들이 합의한 동일 수신 목록으로 마감하도록 조정하세요.

### 서버 검증과 FedAvg

```python
from fedsec.aggregation import fedavg, validate_updates
from fedsec.contracts import RoundContext

context = RoundContext(experiment_id, round_id, model_version, tuple(selected_ids))
validate_updates(global_state, received_updates, context, sample_counts=assigned_sample_counts)
next_state = fedavg(global_state, received_updates, context, sample_counts=assigned_sample_counts)
```

기한 전에 도착한 유효 업데이트만 전달하세요. 선정됐지만 미제출인 클라이언트는 제외하며,
FedAvg 분모는 실제 수신 표본 수 합계입니다. 유효 업데이트가 0개면 오류입니다.
파일 크기 제한, 제출 마감, 재시도는 API 계층에서 처리하세요.
JSON 메타데이터와 바이너리 안의 메타데이터도 일치하는지 확인하세요.
서버 입력에 공격 여부나 악성 ID 목록을 추가할 필요가 없습니다.

### 데이터 분할 교환

이미지 파일을 나눠 전달하면 `train_local()`에 해당 Dataset을 직접 넘깁니다.
모두 같은 공식 CIFAR-10을 갖고 있다면 인덱스 파일을 공유해도 됩니다.
`data.partition_file` 구조:

```json
{
  "format_version": 1,
  "dataset": "CIFAR-10",
  "source_split": "official_train",
  "source_size": 50000,
  "targets_sha256": "공식 학습 레이블 배열 bytes의 SHA-256",
  "num_clients": 10,
  "client_indices": {"0": [0, 10], "1": [1, 11]},
  "validation_indices": []
}
```

위 목록은 구조 설명용으로 생략되어 있습니다. 실제 파일은 모든 클라이언트를 포함하고
원본 학습 데이터를 클라이언트/검증에 중복 없이 전체 배정해야 합니다.
`make_iid_partition()`을 참고하거나 재사용할 수 있고, 외부 분할은 비IID도 허용합니다.

## 2번 · 방어와 DLG 담당자

클리핑은 delta에 적용한 뒤 FedAvg합니다.
좌표별 중앙값은 중앙값 delta로 W를 갱신하며 FedAvg를 추가로 하지 않습니다.
업데이트 자료형을 유지하려면 `update.with_delta(new_delta)`를 사용하세요.

독립 실행기 확장 함수:

```python
def aggregate_with_defense(global_state, updates, context, defense_config):
    # updates는 검증된 제출값. 공격자 정답 목록은 전달되지 않음.
    # 방어/집계 후 갱신된 state_dict 반환.
    ...

# simulate(config, mode="compare", aggregation_fn=aggregate_with_defense)
```

DLG의 동일 모델 복원:

```python
from fedsec.artifacts import load_checkpoint
from fedsec.config import ModelConfig
from fedsec.model import build_model

checkpoint = load_checkpoint("received/global.pt")
model = build_model(ModelConfig(**checkpoint["model_config"]), checkpoint["model_state"])
preprocessing = checkpoint["preprocessing"]
```

시나리오대로 서버 → 매니저 → 클라이언트 순서로 같은 W 파일을 전달하세요.
버전/해시/전처리를 맞춰 단일 이미지 그래디언트를 계산하고, 노이즈는 클라이언트에서 한 번 적용합니다.
그래디언트 이름/순서와 서버의 레이블 접근 조건은 2번·3번이 함께 정하세요.
매니저는 원본과 복원 이미지를 같은 [0,1] RGB 공간으로 맞춘 뒤 MSE를 계산합니다.
초안의 서버 MSE 필드는 매니저 계산값을 기록하거나 비워 두는 방식으로 조정하세요.
