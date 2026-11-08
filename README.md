# FedSecuSimulator

연합학습 보안 시뮬레이션 및 공격·방어 결과 분석 프로젝트입니다.

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

## DLG 실험

웹 UI에서 실험 유형을 `DLG 실험 (정상 연합학습 후 공격)`으로 설정합니다.
`DLG 사전 정상 연합학습 라운드 수`만큼 정상 라운드를 실행한 뒤 DLG 라운드를 시작합니다.

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
```

DLG 결과에는 원본 이미지, 복원 이미지, MSE, 방어 설정, 노이즈 sigma가 포함됩니다.

## 팀 구현 로직 교체

실제 모델, 학습, FedAvg, 평가, 공격/방어, DLG 구현을 교체하는 함수와 payload 형식은
[TEAM_IMPLEMENTATION.md](TEAM_IMPLEMENTATION.md)에 정리되어 있습니다.
