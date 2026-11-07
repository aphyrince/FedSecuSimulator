# FedSecuSimulator

연합학습 보안 시뮬레이션 팀 프로젝트의 **1번 담당 코드**입니다.
CIFAR-10 공통 모델, 로컬 학습, FedAvg, 내부자 협력 공격과 독립 실행기를 제공합니다.
설정 파일을 읽거나 모듈을 import하는 것만으로 학습이 시작되지 않습니다.

## 역할과 구현 범위

| 담당 | 범위 | 연결 |
| --- | --- | --- |
| 1번 · 김시형 | CNN, 가상 클라이언트, 로컬 학습, FedAvg, 협력 공격, 정상·공격 비교 | 이 저장소 |
| 2번 | 클리핑, 좌표별 중앙값, DLG, 가우시안 노이즈 | 공통 모델/체크포인트와 집계 확장 함수 사용 |
| 3번 | Flask/REST, 프로세스, 데이터 분할/전송, 참여·지연·기한 관리, 로그·시각화 | 알고리즘 함수를 API 작업에서 호출 |

독립 실행기는 **한 프로세스에서 가상 클라이언트를 순서대로** 처리합니다.
실제 10개 프로세스, HTTP 통신, 타임아웃, 지연 실험을 구현한 것으로 간주하지 않습니다.
2번 알고리즘은 아직 포함하지 않았습니다. 방어 활성화 후 구현을 연결하지 않으면 실행 전에 오류를 냅니다.

## 설치

프로젝트 루트의 PowerShell에서 실행합니다. Python 3.10 이상을 사용합니다.
requirements.txt는 기존 환경의 PyTorch/torchvision 조합으로 고정했습니다.

```powershell
python -m venv venv
.\venv\Scripts\python.exe -m pip install -r requirements.txt
```

기존 venv와 CIFAR-10 원본이 있으면 재사용할 수 있습니다.

## 먼저 설정만 확인하기 · 학습 없음

```powershell
.\venv\Scripts\python.exe -m fedsec validate --config configs/experiment.json --mode compare
```

또는 실제 실행 명령에 `--dry-run`을 붙입니다.

```powershell
.\venv\Scripts\python.exe -m fedsec simulate --config configs/development.json --mode compare --dry-run
```

두 명령 모두 설정과 참여 일정을 출력합니다. 데이터 다운로드·학습·결과 파일 생성을 하지 않습니다.
형식/값/설정 조합을 검사하며, 실제 데이터나 외부 체크포인트 파일까지 읽는 검사는 아닙니다.

## 설정 파일

| 파일 | 용도 |
| --- | --- |
| `configs/experiment.json` | 시나리오 기본 예시: 10개, 20라운드, 악성 2개, 배율 -5 |
| `configs/development.json` | 연결 확인용 예시: 1라운드, 클라이언트당 20장, 검증 100장 |
| `configs/local.example.json` | 개인 PC 예시. `local.json`으로 복사해 수정 가능 |

모든 수치는 조정 가능한 시작 예시이며, 성능 검증을 거쳐 선정한 값이 아닙니다.
`extends`는 설정을 **객체별로 병합**합니다. 배열은 전체 교체합니다.
개인 파일 `configs/local.json`과 `configs/*.local.json`은 Git에서 제외됩니다.
데이터·출력·체크포인트 경로는 절대경로 또는 **프로젝트 루트 기준 상대경로**입니다.
`extends` 경로만 현재 설정 파일 기준입니다.

| 설정 | 의미 |
| --- | --- |
| `num_clients`, `num_rounds` | 전체 클라이언트 수와 라운드 수 |
| `participation_rate` | 독립 실행 참여율. `ceil(전체 수 × 참여율)`, 최소 1개 |
| `participation_schedule` | `{"1": [0, 2, 7]}`처럼 특정 라운드 참여 ID 지정 |
| `training` | `local_epochs`, `batch_size`, `learning_rate`, SGD/Adam 옵션 |
| `model` | 합성곱 채널 수, 완전연결 크기. 모든 참여자가 같은 설정 사용 |
| `data.partition_file` | 3번이 만든 분할 JSON. 지정 시 해당 배정 사용 |
| `data.validation_size` | 외부 분할이 없을 때 학습셋에서 떼어낼 검증 표본 수 |
| `data.normalization_*` | RGB 전처리. 서버·클라이언트·DLG에서 동일하게 사용 |
| `evaluation.split` | `test` 또는 `validation` |
| `attack.malicious_clients`, `scale` | 악성 ID 목록과 변조 배율 |
| `attack.min_colluders` | 최소 협력자 수 |
| `attack.missing_peer_policy` | `skip_attack`, `available_only`, `error` |
| `initial_checkpoint` | 기존 공통 체크포인트에서 출발. null이면 새 모델 |
| `runtime` | CPU/CUDA/auto, worker·스레드 수, 경로, 결정론 설정 |

공식 시나리오 설정은 학습 50,000장을 10개에 배정하므로 **각 5,000장**이고,
공식 테스트 10,000장으로 매 라운드 평가합니다.
개발 설정은 학습 40,000장·검증 10,000장으로 분리한 뒤 표본 제한을 적용합니다.
테스트 결과를 보고 학습률 등을 고르지 말고 개발/검증 설정에서 조건을 정하세요.

`num_clients`를 바꾸면 악성 ID도 확인하세요. 설정의 오타·알 수 없는 키·중복 키·NaN/Infinity는 오류로 처리합니다.

## 사용자가 직접 실행하기 · 아래부터 학습 실행

원본이 없을 때만 데이터 준비 명령을 실행합니다. 이 명령에는 학습이 없습니다.

```powershell
.\venv\Scripts\python.exe -m fedsec download-data --config configs/experiment.json
```

작은 표본으로 연결을 먼저 확인하려면:

```powershell
.\venv\Scripts\python.exe -m fedsec simulate --config configs/development.json --mode compare --run-name dev_001
```

본 시나리오의 정상 연합학습:

```powershell
.\venv\Scripts\python.exe -m fedsec simulate --config configs/experiment.json --mode normal --run-name normal_001
```

정상·협력 공격 비교:

```powershell
.\venv\Scripts\python.exe -m fedsec simulate --config configs/experiment.json --mode compare --run-name compare_001
```

`compare`는 정상 1회와 공격 1회를 순차 실행합니다. 같은 초기 가중치·분할·참여 일정과
클라이언트/라운드별 시드를 사용합니다. 공격 이후 전역 모델이 달라지는 것은 의도된 차이입니다.
기존 결과를 덮어쓰지 않으므로 다시 실행할 때 새 `--run-name`을 지정하세요.
실패 시 완료된 라운드 결과는 보존합니다. 중단 재개 명령은 제공하지 않습니다.

## 계산 규칙

1. 같은 라운드의 모든 참여자는 같은 W의 독립 복사본에서 출발합니다.
2. `delta = W_local - W_start`. `num_samples`는 실제 표본 수이며 epoch 수를 곱하지 않습니다.
3. FedAvg: `W_next = W_start + Σ(n_k / Σn_j × delta_k)`.
4. 협력자는 변조 전 delta를 교환하고 **단순평균 × scale**을 각자의 제출 delta로 사용합니다.
5. 실제 참여한 악성 클라이언트만 협력 대상으로 봅니다.
6. `skip_attack`은 참여 협력자 업데이트가 누락되면 수신한 정상 값을 그대로 제출합니다.
   `available_only`는 수신한 값만 사용하되 최소 협력자 수를 충족해야 합니다. `error`는 누락 시 예외입니다.
7. 서버 집계 함수는 악성 ID를 받지 않습니다. 공격 기록은 독립 실행기의 매니저용 로그에 남깁니다.
8. 실험/라운드/모델 버전, 시작 모델 해시, 이름·모양·자료형·NaN/Inf, 중복 ID를 검사합니다.
   실제 서버는 선정 여부와 데이터 배정 기록의 표본 수도 검증해야 합니다.

## 파일과 팀원 연결

| 코드 | 기능 |
| --- | --- |
| `fedsec/config.py` | JSON 설정과 유효성 검사 |
| `fedsec/model.py` | 공유 CNN |
| `fedsec/training.py` | 로컬 학습과 분류 평가 |
| `fedsec/aggregation.py` | 업데이트 검사와 FedAvg |
| `fedsec/attack.py` | 협력 공격과 클라이언트별 수신함 |
| `fedsec/contracts.py` | `/train`, 업데이트, 라운드 자료형 |
| `fedsec/artifacts.py` | 바이너리 업데이트와 전역 모델 체크포인트 |
| `fedsec/integration.py` | API 요청과 체크포인트를 학습에 연결 |
| `fedsec/data.py` | 독립 실행용 분할과 데이터 로딩 |
| `fedsec/simulation.py` | 정상/공격 실행과 비교 |

상세 연결 예제는 [docs/INTEGRATION.md](docs/INTEGRATION.md)를 참고하세요.

학습 함수 `train_local()`은 최종 `training_config` 하나로 epoch·배치·학습률·옵티마이저를 결정합니다.
API 연결에서는 `integration.resolve_training_config()`가 기본 설정과 서버 요청값을 합칩니다.
독립 실행에서는 JSON의 `training` 설정을 그대로 사용합니다. 실험/라운드/모델 버전은 `context`로 전달합니다.

`simulation.py`는 아래 순서로 읽으면 됩니다.

| 함수 | 역할 |
| --- | --- |
| `simulate()` | 실행 조건 확인, 라운드 반복, 정상·공격 비교와 요약 저장 |
| `prepare_experiment()` | 데이터·초기 모델 준비와 재현용 파일 저장 |
| `run_round()` | 한 라운드의 로컬 학습 → 공격 → 집계 → 평가와 결과 저장 |

`PreparedExperiment`가 준비한 데이터와 초기 모델을 두 비교 조건에 공유합니다.

출력은 `outputs/<run-name>/`에 저장됩니다.
공유 `initial.pt`, `config.json`, `partition.json`, `used_indices.json`,
조건별 `events.jsonl`·`metrics.csv`·`summary.json`, 라운드별 `global.pt`·`result.json`·로컬 이력·정상/제출 업데이트,
비교 시 `comparison.json`이 생성됩니다. 정확도와 정확도 차이는 0~1 비율입니다.
체크포인트는 동일 모델과 전처리를 복원할 수 있어 2번의 DLG에 전달할 수 있습니다.

## Git에 올릴 파일

코드, 테스트, requirements.txt, 공통/개발/예제 설정, 문서를 공유합니다.
가상환경·원본 데이터·생성 결과·기존 분할 캐시·개인 설정·모델 바이너리(`.pt`, `.pth`, `.ckpt`)는 제외합니다.
`.env`와 `.env.*`도 제외하며 `.env.example` 또는 `.env.*.example`은 공유할 수 있습니다.
모델은 Git 대신 파일로 전달하고 동일 버전/해시를 확인하세요.

## 학습 없는 검증

```powershell
.\venv\Scripts\python.exe -m unittest discover -s tests -v
```

텐서 산술·설정·분할·파일 형식과 가짜 학습/평가 함수를 사용한 실행 흐름만 검사합니다.
실제 모델의 forward/backward, 옵티마이저 학습, CIFAR-10 다운로드를 수행하지 않습니다.
실제 수렴 성능, 공격 효과, GPU 동작, Flask 통신은 사용자가 실행한 뒤 확인해야 합니다.
