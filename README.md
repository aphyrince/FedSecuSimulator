# FedSecuSimulator

연합학습 보안 시뮬레이션 및 공격·방어 결과 분석 프로젝트입니다.

## 1번 담당: Flask 없이 독립 실행

네 담당 알고리즘의 원본은 `fedsec/`에 있습니다. 서버/클라이언트는 이 함수를 재사용하며,
기존 REST 요청·응답 형태를 유지하는 변환 코드는 `common/`, `clients/services/`, `server/services/`에 남습니다.
`fedsec`는 Flask나 매니저를 import하지 않습니다.

| 파일 | 역할 |
| --- | --- |
| `fedsec/experiment.py` | `run_comparison_experiment()`로 정상 학습 → 공격 학습 → 비교 관리 |
| `fedsec/training.py` | 모델 복사 후 로컬 학습, 테스트 정확도·loss 평가 |
| `fedsec/aggregation.py` | Tensor 기반 FedAvg |
| `fedsec/attack.py` | 정상 업데이트 단순평균 × 공격 배율 |
| `fedsec/model.py` | 모든 실행 방식에서 공유하는 SmallCNN |
| `fedsec/config.py` | 학습·실험 설정 자료형 |
| `fedsec/contracts.py` | 학습 결과·평가 결과·클라이언트 업데이트 자료형 |
| `fedsec/tensors.py` | 가중치 검증/복사, delta 추출, 모델 해시 |
| `fedsec/data.py` | 이미 정해진 CIFAR-10 배정 범위 로딩; 데이터 분할 정책은 구현하지 않음 |
| `fedsec/comparison.py` | 공통 조건 검사 및 정확도 차이 비교 |
| `fedsec/__main__.py` | 독립 실행 명령 및 파일 저장 |

프로젝트 루트의 새 터미널에서 사용자가 직접 실행합니다. 서버를 켜거나 웹에서 데이터를 분배할 필요는 없습니다.
`data/raw/cifar-10-batches-py/`에 공식 CIFAR-10 파일이 있어야 합니다.

```powershell
.\.venv\Scripts\python.exe -m fedsec --config configs/fedsec.example.json
```

**이 명령은 실제 학습을 실행합니다.** 예제는 10개 클라이언트, 악성 ID 2·7, 1라운드,
클라이언트당 20개 학습 표본, 테스트 100개인 연결 확인용 설정입니다. 본 실험 성능으로 보고하지 마세요.
본 실험에서는 설정 파일을 복사해 라운드/학습 조건을 정하고 두 표본 제한을 `null`로 설정할 수 있습니다.
예제의 데이터 배정표는 각 훈련 배치를 5,000개씩 둘로 나눈 고정 배정입니다. 팀원의 분할 결과와 같다고 가정하지 마세요.
팀원과 같은 데이터를 사용하려면 JSON의 `datasets` 배정 범위를 팀원의 결과에 맞추면 됩니다.

독립 실행은 한 프로세스에서 클라이언트를 순차 학습하고, 전원 참여·지연 없음·방어 없음으로 비교합니다.
참여율/지연/방어 조건을 포함한 시스템 실험은 기존 매니저를 사용합니다.
재실행 시에는 새 ID를 지정합니다.

```powershell
.\.venv\Scripts\python.exe -m fedsec --config configs/fedsec.example.json --experiment-id standalone_check_002
```

결과는 `results/standalone/<ID>/`의 `normal/`, `attack/`, `comparison.json`에 저장됩니다.
각 실험은 `results.json`, `final_model.pt`, 라운드별 JSON을 남깁니다.
중단 시 이미 완료된 라운드 JSON이 남고, 두 실험 모두 완료된 경우에만 최종 결과와 비교 파일을 저장합니다.
기존 결과 폴더는 덮어쓰지 않습니다.

Python에서 사용할 때의 진입점도 하나입니다.

```python
import json
from fedsec import ExperimentSettings, run_comparison_experiment
from fedsec.data import load_datasets

with open("configs/fedsec.example.json", encoding="utf-8") as file:
    payload = json.load(file)
config = ExperimentSettings.from_dict(payload["experiment"])
clients, test = load_datasets(payload["datasets"], config.training)
result = run_comparison_experiment(config, clients, test, experiment_id=payload["experiment_id"])
print(result.comparison["final"])
```

Python 함수는 결과를 반환하고 파일 저장은 CLI가 맡습니다. 정상·공격 학습은 동일한 초기 가중치의
독립 복사본으로 시작합니다. 데이터 배정과 설정도 같게 유지합니다.
`common`의 기존 import 경로는 호환용으로 유지하므로 팀원의 호출 경로를 바꿀 필요는 없습니다.

## 현재 master의 실제 학습 구현

1번 담당의 CNN·로컬 학습·FedAvg·실제 분류 평가·내부자 협력 공격을 현재 서버/클라이언트에 연결했습니다.
모델과 업데이트는 기존 JSON API 키를 유지하며, `weights`/`update.values`에 파라미터별 중첩 실수 배열을 넣습니다.
토치 모델 구조와 전처리는 `fedsec/model.py`에서 정의하고 `common/model.py`가 REST 형식으로 연결합니다. torchvision은 사용하지 않습니다.

| 파일/함수 | 역할 |
| --- | --- |
| `common/model.py` | 공통 SmallCNN, JSON ↔ 텐서, 초기 모델과 체크포인트 |
| `common/data.py` | 매니저가 배정한 CIFAR-10 파일의 `[start,end)` 범위 로딩 |
| `clients/services/local_training.py:train_local_model()` | 같은 시작 모델의 독립 복사본으로 실제 로컬 학습 |
| `server/services/fedavg.py:aggregate_fedavg()` | 실제 표본 수로 가중 평균한 delta를 전역 모델에 더함 |
| `server/services/model_evaluation.py:evaluate_global_model()` | 실제 예측 정확도와 표본 가중 CrossEntropy loss |
| `clients/services/collusion.py:collude_updates()` | 정상 delta의 단순평균 × 공격 배율, 제출 모델에도 반영 |
| `common/comparison.py:compare_accuracy_results()` | 저장된 정상·공격 실험을 공통 조건 검사 후 비교 |

아래 조작은 사용자가 직접 실행합니다. 파일 import와 테스트는 실제 학습을 시작하지 않습니다.

1. Python 3.13.3의 `.venv`에 `requirements.txt`를 설치합니다.
2. `exp_manager/.env`의 `SOURCE_DIR`를 실제 데이터 폴더로 설정합니다. 예제는 `.env.example`입니다.
   기존 데이터가 `data/raw/cifar-10-batches-py/`에 있으면 `SOURCE_DIR=data/raw`를 사용합니다.
3. `python exp_manager/app.py` 대신 아래 명령으로 정확한 가상환경을 지정해 실행할 수 있습니다.

```powershell
.\.venv\Scripts\python.exe exp_manager\app.py
```

4. `http://127.0.0.1:8000`에서 **훈련/테스트 데이터 범위 분배**를 먼저 실행합니다.
5. 첫 연결 확인은 새 실험 ID, 1라운드, 로컬 1 epoch, 학습 표본 제한 20, 평가 제한 100으로 진행할 수 있습니다.
   이 수치는 연결 확인용이며 본 실험 성능으로 보고하지 않습니다.
6. 기준 비교는 참여율 1, 지연 없음으로 시작합니다. 정상/공격 조건에서 시드·모델·데이터 배정·학습 설정을 같게 둡니다.
   상단 공격/방어 옵션을 변경한 뒤 **라운드 설정표 생성 → 설정 저장**으로 각 라운드에도 반영하세요.
7. 정상은 `normal_001`, 공격은 `attack_001`처럼 서로 다른 ID로 저장합니다. 결과가 있는 ID의 재실행은 차단합니다.

웹 화면에서 epoch·배치·학습률·CPU/CUDA·제출 제한 시간·표본 제한을 설정할 수 있습니다.
내부자 협력 공격을 켜면 공격 설정에 업데이트 배율 입력란이 나타납니다(기본값 `-5`).
저장한 배율은 실험 관리자가 각 공격 라운드의 악성 클라이언트에 전달합니다.
전체 표본을 사용할 때는 표본 제한을 비웁니다. 모델 채널/완전연결 크기와 옵티마이저 세부 옵션은
`POST /api/experiment/config`의 `model_config`, `optimizer`, `momentum`, `weight_decay`로 전달할 수 있습니다.
기본값은 SGD, lr=0.001, batch=32, local_epochs=2, CPU이며 성능 튜닝으로 확정한 값이 아닙니다.
데이터는 기존 매니저의 범위 분배를 그대로 사용합니다. 클라이언트별 표본 수는 균등하지 않을 수 있고,
Non-IID 생성은 3번 담당자의 분배기 연결이 필요합니다. 화면에서 분포 옵션을 선택하는 것만으로 새 분할이 생성되지는 않습니다.
실제 로컬 학습은 원래의 3초 요청 시간보다 길어지므로 라운드 제출 제한을 기본 900초로 두었습니다.
실행 환경에 맞춰 조정하세요. 프로세스 초기 준비는 최대 60초 기다립니다.

### 매니저 API를 통한 정상 학습 → 공격 학습 → 정확도 비교

서버 연동용 진입 함수는 `exp_manager/services/comparison_runner.py`의 `run_comparison_experiment()`입니다.
위 `fedsec`의 독립 실행과 달리 실행 중인 매니저가 필요합니다.
서버/클라이언트 실행과 데이터 분배를 마치고 웹에서 학습 설정을 저장한 다음,
VS Code의 **새 터미널**에서 실행합니다. 아래 명령은 실제 학습을 두 번 실행합니다.

```powershell
.\.venv\Scripts\python.exe -m exp_manager.services.comparison_runner compare_001
```

현재 매니저의 설정을 한 번 읽어 정상 실험 `compare_001_normal`을 완료한 뒤,
같은 시드·모델 구조·학습 설정·데이터 배정으로 `compare_001_attack`을 새 초기 모델에서 실행합니다.
라운드별 설정의 공격 여부도 각각 강제 적용하고, 두 실험 모두 방어를 꺼서 FedAvg 기준으로 비교합니다.
악성 후보와 공격 배율은 저장된 설정을 사용합니다. FedAvg는 각 라운드 안에서 서버가 호출합니다.
비교 중에는 웹에서 다른 실험을 시작하거나 설정/데이터를 변경하지 마세요.

Python에서 직접 호출하거나, CLI의 `--config 설정파일.json`으로 설정 일부를 덮어쓸 수도 있습니다.

```python
from exp_manager.services.comparison_runner import run_comparison_experiment

result = run_comparison_experiment({
    "experiment_id": "compare_002",
    "num_rounds": 20,
    "seed": 42,
    "local_epochs": 2,
    "learning_rate": 0.001,
    "participation_rate": 1.0,
    "malicious_clients": [2, 7],
    "attack_scale": -5.0,
    "delay_clients": [],
    "round_configs": [],  # 웹의 라운드별 덮어쓰기를 사용하지 않고 위 공통 설정 사용
})
print(result["final"])
```

두 실험의 원본 결과는 각각 `results/<실험 ID>/`에,
비교표는 `results/compare_001_comparison.json`에 저장됩니다.
기존 결과를 덮어쓰지 않으므로 다시 실행할 때는 새 ID를 사용하세요.
실패·중단 시 다음 실험으로 넘어가지 않으며, 조건이 다른 결과는 비교 파일로 저장하지 않습니다.
Ctrl+C 또는 대기 시간 초과 시 현재 실험에 중단을 요청합니다. 진행 중인 라운드는 바로 끝나지 않을 수 있습니다.
네트워크 오류로 연결이 끊기면 매니저는 계속 실행 중일 수 있으니 웹에서 상태를 확인하세요.
참여율이 낮아 협력 공격이 실제 적용되지 않은 라운드는 `attack_applied=false`로 표시됩니다.

| 함수 | 담당 기능 |
| --- | --- |
| `run_comparison_experiment()` | 정상·공격 실험을 순서대로 실행하고 비교 결과 저장 |
| `run_federated_learning()` | 매니저 API로 한 실험을 시작하고 전체 라운드 완료 대기 |
| `train_local_model()` | 각 클라이언트의 CIFAR-10 로컬 학습 |
| `collude_updates()` | 악성 클라이언트 정상 delta 평균에 공격 배율 적용 |
| `aggregate_fedavg()` | 클라이언트 표본 수에 따른 가중 평균으로 전역 모델 갱신 |
| `evaluate_global_model()` | 테스트 데이터의 정확도와 loss 계산 |
| `compare_accuracy_results()` | 동일 조건 여부 검사 후 라운드별 정확도 및 하락 폭 비교 |

### 저장된 정상·공격 정확도 비교 · 학습 실행 없음

두 실험이 완료되면 실행합니다.

```powershell
.\.venv\Scripts\python.exe -m common.comparison results/normal_001/results.json results/attack_001/results.json
```

같은 초기 모델 해시·학습 설정·실제 사용 데이터·참여 및 제출 클라이언트를 검사합니다.
기준이 다르거나 예전 가상 평가 결과이면 비교를 거부합니다. 정확도는 0~1, 하락 폭은 비율과 %p로 제공합니다.
참여자 중 악성이 1개뿐이면 협력 공격을 생략하고 정상 delta를 제출합니다. `attack_applied`도 확인하세요.

### 결과와 검증

매 라운드 전역 모델과 실제 제출 delta는 `results/<ID>/checkpoints/`와 `updates/`에 `.pt`로 저장합니다.
클라이언트의 변조 전 모델/delta도 `results/<ID>/clients/`에 저장합니다.
JSON/CSV/API 이력은 무거운 가중치 배열 대신 요약과 체크포인트 경로를 기록합니다.
`GET /api/model/global`은 현재 전체 모델을, 클라이언트 extracted-update API는 저장한 delta를 반환합니다.
학습·집계·평가 실패 시 가짜 정확도를 만들지 않고 오류를 반환합니다.

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_master_*.py" -v
```

검증은 작은 고정 텐서·로컬 합성 데이터 파일·가짜 학습/평가 함수로 수행하며 실제 CNN 학습과 서버 프로세스를 실행하지 않습니다.
실제 데이터에서의 수렴, 공격 효과, GPU 성능은 사용자의 실행으로 확인해야 합니다.

**DLG는 2번 담당자의 연결 작업이 남아 있습니다.** 기존 가상 DLG는 실제 CNN과 호환되지 않아
현재 실제 모델의 DLG 요청은 명시적 오류로 차단합니다. `common.model.build_model()`과
`load_model_checkpoint()`를 이용해 같은 모델·가중치·전처리를 사용하는 DLG로 연결하세요.

## 실행

실험 관리자만 실행하면 중앙 서버와 클라이언트 10개를 함께 시작합니다.

```powershell
.\.venv\Scripts\python.exe exp_manager\app.py
```

웹 UI: http://127.0.0.1:8000

- **전체 프로세스 시작**: 중앙 서버와 클라이언트 프로세스 시작 또는 재시작
- **전체 프로세스 종료**: 자식 프로세스 종료 REST API 호출
- **실험 관리자 종료**: 자식 프로세스를 종료한 뒤 관리자 종료
- **실험 시작/중단**: 전체 연합학습 또는 DLG 실험 실행

개별 프로세스 실행:

```powershell
.\.venv\Scripts\python.exe server\app.py
.\.venv\Scripts\python.exe clients\app.py --client-id 0 --port 5101
```

기본 포트:

- 실험 관리자: `8000`
- 중앙 서버: `5000`
- 클라이언트: `5101`~`5110`

## 실험 관리자 API

- `GET /api/experiment/config`: 설정 조회
- `POST /api/experiment/config`: 설정 변경
- `POST /api/experiment/start`: 실험 시작
- `POST /api/experiment/stop`: 실험 중단
- `GET /api/experiment/status`: 진행 상태 조회
- `POST /api/experiment/reset`: 상태 초기화
- `POST /api/results/round`: 중앙 서버의 정상 라운드 결과 수신
- `POST /api/dlg/results`: 중앙 서버의 DLG 복원 결과 수신

## CIFAR-10 데이터 설정

프로젝트 루트 기준으로 CIFAR-10 원본 배치를 다음 위치에 둡니다.

```text
source/
└── cifar-10-batches-py/
    ├── data_batch_1
    ├── data_batch_2
    ├── data_batch_3
    ├── data_batch_4
    ├── data_batch_5
    ├── test_batch
    └── batches.meta
```

`exp_manager/.env`에서 데이터 위치를 설정합니다.

```dotenv
SOURCE_DIR=source
# 아래 값을 비워두면 위 공식 CIFAR-10 폴더의 5개 훈련 배치와 test_batch를 사용합니다.
TRAIN_DATA_FILE=
TEST_DATA_FILE=
CIFAR_TEST_RATIO=0.2
DLG_IMAGE_PATH=data/dlg/sample.json
DLG_LABEL_PATH=data/dlg/sample_label.txt
```

다른 폴더에 배치 파일을 둔 경우 `SOURCE_DIR`는 프로젝트 루트 기준 상대 경로로 입력하고,
`TRAIN_DATA_FILE`, `TEST_DATA_FILE`에는 `SOURCE_DIR` 기준 상대 파일명을 쉼표로 구분해서 지정합니다.

예:

```dotenv
SOURCE_DIR=source
TRAIN_DATA_FILE=cifar_custom/train_batch_1,cifar_custom/train_batch_2
TEST_DATA_FILE=cifar_custom/test_batch
```

CIFAR-10은 기본적으로 훈련 배치와 테스트 배치가 이미 분리되어 있으므로
`data_batch_1`~`data_batch_5`와 `test_batch`를 미리 나누어 놓아야 합니다.
`TRAIN_DATA_FILE`과 `TEST_DATA_FILE`을 비워두면 위 공식 폴더 구조를 자동으로 사용합니다.
`CIFAR_TEST_RATIO`는 파일을 명시하지 않았을 때의 자동 탐색에 사용되는 설정이며,
현재 데이터 분배 과정은 훈련/테스트 배치 파일을 별도로 읽어 분배하므로
일반 훈련 데이터에서 이 비율로 자동 분할하지는 않습니다.

실험 관리자에서 데이터 분배 API를 호출합니다.

- `GET /api/data/status`
- `POST /api/data/distribute`

분배 요청 예시:

```powershell
Invoke-RestMethod -Method Post `
  -Uri http://127.0.0.1:8000/api/data/distribute `
  -ContentType application/json `
  -Body '{"seed":42}'
```

## 일반 연합학습 API

- 중앙 서버 라운드 설정: `POST /api/round/config`
- 라운드 시작: `POST /api/rounds/<round_id>/start`
- 업데이트 제출: `POST /api/rounds/<round_id>/updates`
- 제출 상태 조회: `GET /api/rounds/<round_id>/updates/status`
- 현재 전역 모델: `GET /api/model/global`
- 전역 모델 이력: `GET /api/model/history`
- 라운드 평가: `GET /api/evaluation/<round_id>`

## DLG 연결 예정 인터페이스

웹 UI에서 `DLG 실험` 탭을 선택합니다. 기본 동작은 `DLG 사전 정상 연합학습 라운드 수`만큼 정상 라운드를 실행한 뒤 DLG 라운드를 시작하는 것입니다.

이미 학습된 모델을 사용하려면 `사전 정상학습 전역모델을 사용`을 선택하고 JSON 체크포인트를 업로드하거나 붙여넣습니다. CSV/텍스트 입력은 현재 `small_cnn_v1` 모델의 `state_dict` 파라미터 순서와 일치하는 숫자 가중치 목록이어야 합니다. 설정 저장 시 모델 구조와 가중치를 검증하며, DLG는 사전 학습 라운드 없이 해당 체크포인트로 시작합니다.

내부자 협력공격/정상 연합학습 실험이 완료되면 최종 전역 모델이 아래 파일로 별도 저장됩니다.

```text
results/<experiment_id>/global_model_for_dlg.json
```

이 파일을 DLG 실험에서 재사용하려면 DLG 탭에서 `사전 정상학습 전역모델을 사용`을 선택하고, 입력 형식을 `JSON 체크포인트`로 둔 뒤 위 파일을 선택하고 설정을 저장합니다. 저장한 체크포인트로 바로 DLG를 실행하며 사전 연합학습 라운드는 생략됩니다.

DLG 관련 API:

- 참가 클라이언트 지정: `POST /api/dlg/participant`
- 클라이언트 DLG 시작: `POST /api/dlg/start`
- 그래디언트 제출: `POST /api/dlg/gradient`
- 실험 관리자 결과 수신: `POST /api/dlg/results`

DLG 방어가 활성화된 경우 클라이언트가 계산한 그래디언트에 가우시안 노이즈를 추가합니다.
노이즈 크기는 웹 UI의 `DLG 가우시안 노이즈 σ`에서 설정합니다.

## 실험 결과 저장

결과는 다음 위치에 저장됩니다.

```text
results/<experiment_id>/rounds/round_<번호>.json
results/<experiment_id>/rounds.csv
results/<experiment_id>/results.json
results/<experiment_id>/accuracy.csv
results/<experiment_id>/global_model_for_dlg.json
```

연합학습 완료 시 `global_model_for_dlg.json`이 생성됩니다. DLG 결과에는 원본 이미지, 복원 이미지, MSE, 방어 설정, 노이즈 sigma가 포함됩니다.

## 팀 구현 로직 교체

구현 함수와 payload 형식, DLG 등 남은 연결 지점은
[TEAM_IMPLEMENTATION.md](TEAM_IMPLEMENTATION.md)에 정리되어 있습니다.
