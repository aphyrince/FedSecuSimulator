# FedSecuSimulator 프로젝트 가이드

프로젝트 설치부터 실험 실행, 결과 확인 및 재실행까지 설명합니다. 빠른 개요는 [README](../README.md)를 참고하세요.

## 목차

- [구성](#구성)
- [설치 및 데이터 준비](#설치-및-데이터-준비)
- [실행 순서](#실행-순서)
- [실험 모드](#실험-모드)
- [결과와 재현성](#결과와-재현성)
- [주요 API](#주요-api)
- [Flask 없는 독립 실행](#flask-없는-독립-실행)
- [테스트 및 유의사항](#테스트-및-유의사항)

## 구성

| 경로 | 역할 |
| --- | --- |
| `fedsec/` | Flask와 독립적인 모델, 로컬 학습, FedAvg, 내부자 공격 및 결과 비교 |
| `common/` | 모델·데이터·체크포인트·텐서 변환 등 서버와 클라이언트의 공통 기능 |
| `clients/` | 클라이언트 Flask 앱, 로컬 학습, 공격 및 DLG 그래디언트 계산 |
| `server/` | 중앙 서버 Flask 앱, 라운드 실행, 집계, 평가 및 DLG 복원 |
| `exp_manager/` | 실험 관리자 Flask 앱, 웹 UI, 데이터 분배, 프로세스와 실험 관리 |
| `results/` | 실험 로그, 체크포인트, 업데이트, 비교 보고서 및 연속 실험 결과 |

매니저 기반 실행에서는 서버와 클라이언트가 REST API로 통신합니다. `fedsec/`는 이 프로세스 구성과 별도로
학습 및 공격 비교를 실행하는 모듈입니다.

## 설치 및 데이터 준비

프로젝트 루트에서 가상환경과 의존성을 준비합니다.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

공식 CIFAR-10 Python 배치를 다음 구조로 둡니다.

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

`exp_manager/.env`를 만들고 경로와 데이터 설정을 지정합니다. 예시는 `exp_manager/.env.example`에 있습니다.

```dotenv
SOURCE_DIR=source
TRAIN_DATA_FILE=
TEST_DATA_FILE=
CIFAR_TEST_RATIO=0.2
DLG_IMAGE_PATH=data/dlg/sample.json
DLG_LABEL_PATH=data/dlg/sample_label.txt
```

`SOURCE_DIR`는 프로젝트 루트 기준 상대 경로입니다. 별도 훈련·테스트 파일을 지정할 경우 각 파일명은
`SOURCE_DIR` 기준 상대 경로이며 쉼표로 구분합니다. 파일 설정이 비어 있으면 CIFAR-10 배치를 자동 탐색합니다.

## 실행 순서

1. 프로젝트 루트에서 실험 관리자를 시작합니다.

   ```powershell
   .\.venv\Scripts\python.exe exp_manager\app.py
   ```

2. [http://127.0.0.1:8000](http://127.0.0.1:8000)을 열고 데이터 분배를 실행합니다.
3. 실험 모드와 설정을 확인한 뒤 실행합니다.

관리자는 중앙 서버와 클라이언트 10개를 시작하고, 종료 시 자식 프로세스 종료 REST API를 호출합니다.
기본 포트는 관리자 `8000`, 서버 `5000`, 클라이언트 `5101`~`5110`입니다. 서버 또는 클라이언트를
개별 실행할 때는 다음 명령을 사용합니다.

```powershell
.\.venv\Scripts\python.exe server\app.py
.\.venv\Scripts\python.exe clients\app.py --client-id 0 --port 5101
```

데이터 분배는 IID의 시드 기반 분할 또는 실제 CIFAR-10 레이블 기반 Non-IID 분할을 생성합니다.
클라이언트별 클래스·표본 수와 데이터 파일 해시가 매니페스트에 기록되고, 분할 정보는
`results/data_splits/`에 저장됩니다.

연결 확인에는 짧은 라운드, 작은 로컬 epoch, 낮은 표본 제한을 사용할 수 있지만, 제한 표본 결과를 본 실험
성능으로 해석하지 마세요. 기본 학습 설정은 SGD, 학습률 `0.001`, 배치 크기 `32`, 로컬 epoch `2`, CPU입니다.
이는 성능 튜닝을 보장하지 않으므로 실행 환경에 맞게 조정하세요.

## 실험 모드

### 개별 연합학습 및 내부자 공격

개별 실험에서 라운드 수, 시드, 참여율, 데이터 분포, 로컬 학습 설정, 지연 클라이언트 및 공격·방어를
지정합니다. 라운드별 설정을 사용할 경우 설정표를 생성하고 저장해야 적용됩니다. 기존 결과와 겹치지 않는
실험 ID를 사용하세요.

악성 클라이언트는 업데이트를 조작합니다. 부분 참여로 한 명의 악성 클라이언트만 선택되더라도 공격 배율을
적용합니다. 방어는 노름 클리핑 또는 좌표별 중앙값 집계를 사용할 수 있습니다. 채택 클라이언트가 3개 미만이면
모델을 갱신하지 않고 사유를 기록합니다.

지연 클라이언트는 선택 로그에 포함되지만, 설정된 지연 조건에서 제출 전에 제외되며 선택·제외 시각과 사유가
기록됩니다. 이 실험은 지연 시간만큼 대기시키는 것이 아니라 지연 클라이언트를 명시적으로 제외합니다.

### 연속 연합학습

4개 운영 환경(기준, 지연, 부분 참여, 레이블 편향)과 6개 보안 시나리오(정상, 공격, 노름 클리핑,
좌표별 중앙값, 공격+노름 클리핑, 공격+좌표별 중앙값)를 조합해 24개 실험을 순차 실행합니다.
각 조합의 라운드 수, 참여율, 지연·악성 클라이언트, 데이터 분포, 학습 및 공격·방어 설정을 편집할 수 있습니다.
기본 학습 설정은 개별 실험 화면의 값을 사용합니다.

각 하위 실험은 자동 생성 ID와 `results/<experiment_id>/` 폴더를 사용합니다. 배치 매니페스트는
`results/continuous_runs/<batch_id>/manifest.json`에 저장됩니다.

### DLG 및 연속 DLG

개별 DLG는 정상 연합학습 후 수행하거나, 학습된 전역 모델 체크포인트를 지정해 사전 학습 없이 실행할 수
있습니다. 연속 DLG는 사용자가 제출한 체크포인트, 원본 이미지 및 레이블을 사용해 노이즈 없음·약한 노이즈·
강한 노이즈의 세 조건을 비교합니다. 각 조건은 같은 체크포인트와 복원 초기화 시드를 공유합니다.
입력 이미지는 RGB 32×32로 변환됩니다.

### 통합 실험 (연속 FL + DLG)

통합 실험은 연속 연합학습 24개를 실행한 다음, 각 운영 환경의 정상 FL 체크포인트 4개만 DLG 입력으로
사용합니다. 체크포인트마다 DLG 3조건을 실행하므로 총 12개 DLG, 전체 36개 하위 실험입니다.
각 체크포인트의 세 DLG 조건은 같은 원본 이미지·레이블과 복원 초기화를 공유합니다.

웹 UI에서 24개 FL 조합의 설정과 공통 DLG 입력을 지정할 수 있습니다. 각 통합 실행에는 날짜·시간 기반 ID가
부여되고 다음 구조로 저장됩니다.

```text
results/<integrated_experiment_id>/
├── manifest.json
├── dlg_inputs.json
└── cases/
    └── <experiment_id>/
        ├── experiment_info.json
        ├── results.json
        └── global_model_for_dlg.json
```

DLG 서버는 체크포인트와 클라이언트 그래디언트로 복원합니다. 원본 이미지와 정답 레이블은 공격 입력으로
서버에 전송하지 않고, 실험 관리자에서 MSE 평가와 재실행에 사용합니다. 결과 화면은 실행 중인 통합 실험이
있으면 해당 실행을, 그렇지 않으면 가장 최근 실행을 표시합니다. 다른 실행은 해당 폴더의 `manifest.json`을
화면에 불러와 확인할 수 있습니다.

## 결과와 재현성

일반 연합학습의 결과는 `results/<experiment_id>/`에 저장됩니다.

| 파일·디렉터리 | 내용 |
| --- | --- |
| `rounds/round_<번호>.json`, `rounds.csv` | 라운드·클라이언트 로그 및 평가 |
| `results.json`, `accuracy.csv` | 실험 요약 및 정확도 |
| `checkpoints/`, `updates/`, `clients/` | 라운드 모델, 제출 업데이트 및 클라이언트 상태 |
| `global_model_for_dlg.json` | DLG 입력용 전역 모델 체크포인트 |
| `experiment_info.json` | 실험 설정과 시나리오 |
| `dlg_result.json`, `dlg_inputs.json`, `inputs/model.pt` | DLG 결과와 재실행 입력 |

학습 로그에는 실험·라운드·클라이언트 ID, 공격·방어 설정, 표본 수, 정확도, 업데이트 노름, 채택·제외 사유와
시각이 기록됩니다. 전체 악성 비율과 실제 채택 악성 비율은 구분됩니다. 채택 수가 0이면 채택 악성 비율은
`N/A`이고, 채택 업데이트가 3개 미만이면 모델을 갱신하지 않습니다.

DLG 로그에는 이미지 식별 정보, 노이즈 설정, 복원 반복 횟수, MSE 및 복원 이미지가 포함됩니다.
실험 설정·시드·데이터 분할·필요한 모델 상태도 보존하여 분석할 라운드나 복원 실험을 재실행할 수 있습니다.

정확도 차이 비교는 동일 환경·시드의 최종 테스트 정확도를 사용하며 `%p`로 표시합니다.

- 공격 피해 = 정상 FedAvg − 공격 FedAvg
- 방어 개선 = 공격+방어 − 공격 FedAvg
- 정상 손실 = 정상 FedAvg − 정상+방어

시드별 결과와 평균을 함께 기록하고, 음수 결과도 그대로 보존합니다. 비교 조건이 다르거나 비교할 수 없는
결과는 비교를 거부합니다.

저장된 라운드 재실행 예:

```powershell
Invoke-RestMethod -Method Post http://127.0.0.1:8000/api/experiment/replay `
  -ContentType 'application/json' -Body '{"experiment_id":"fedsec-run","round":3}'
```

## 주요 API

### 실험 관리자

| 메서드 및 경로 | 용도 |
| --- | --- |
| `GET/POST /api/experiment/config` | 실험 설정 조회·수정 |
| `POST /api/experiment/start` | 개별 실험 시작 |
| `POST /api/experiment/continuous/start` | 연속 연합학습 시작 |
| `POST /api/experiment/continuous/dlg/start` | 연속 DLG 시작 |
| `POST /api/experiment/continuous/linked/start` | 통합 실험 시작 |
| `POST /api/experiment/stop` | 실험 중단 요청 |
| `GET /api/experiment/status` | 진행 상태 조회 |
| `POST /api/experiment/reset` | 실험 상태 초기화 |
| `POST /api/experiment/replay` | 저장된 라운드 또는 DLG 실험 재실행 |
| `GET /api/results/continuous` | 연속 실험 통합 결과 조회 |
| `GET /api/data/status` | 데이터 분배 상태 조회 |
| `POST /api/data/distribute` | 클라이언트 데이터 분배 |
| `POST /api/results/round` | 중앙 서버 라운드 결과 수신 |
| `POST /api/dlg/results` | 중앙 서버 DLG 결과 수신 |

### 서버 라운드 및 DLG

- 라운드: `POST /api/round/config`, `POST /api/rounds/<round_id>/start`,
  `POST /api/rounds/<round_id>/updates`, `GET /api/rounds/<round_id>/updates/status`
- 모델·평가: `GET /api/model/global`, `GET /api/model/history`,
  `GET /api/evaluation/<round_id>`
- DLG: `POST /api/dlg/participant`, `POST /api/dlg/start`, `POST /api/dlg/gradient`

## Flask 없는 독립 실행

`configs/fedsec.example.json`을 수정하고 프로젝트 루트에서 실행합니다.

```powershell
.\.venv\Scripts\python.exe -m fedsec --config configs/fedsec.example.json
```

예제는 연결 확인을 위한 작은 설정(1라운드, 제한된 훈련·테스트 표본)입니다. 실제 성능 실험에는 데이터
할당 범위, 라운드 및 표본 설정을 목적에 맞게 조정하고 새 실험 ID를 사용하세요.

```powershell
.\.venv\Scripts\python.exe -m fedsec --config configs/fedsec.example.json --experiment-id standalone_check_002
```

결과는 `results/standalone/<ID>/`의 `normal/`, `attack/`, `comparison.json`에 저장됩니다.
Python API는 `fedsec.ExperimentSettings`, `fedsec.run_comparison_experiment`, `fedsec.data.load_datasets`를
제공합니다. 결과 파일 저장은 CLI가 담당합니다.

비교 러너를 이용한 예:

```powershell
.\.venv\Scripts\python.exe -m exp_manager.services.comparison_runner compare_001
.\.venv\Scripts\python.exe -m exp_manager.services.comparison_runner security_001 --security-seeds 42,43,44
```

비교 보고서는 `results/<ID>_comparison.json` 또는 `results/<ID>_security_comparison.json`에 저장됩니다.

## 테스트 및 유의사항

프로젝트 루트에서 테스트를 실행합니다.

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

테스트는 작은 고정 텐서, 로컬 합성 데이터, 대체된 학습·평가 함수를 사용하며 실제 CNN 학습이나 서버
프로세스를 시작하지 않습니다. 따라서 실제 CIFAR-10 수렴, 공격 효과 및 하드웨어 성능은 별도로 확인해야 합니다.
