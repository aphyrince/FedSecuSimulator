# FedSecuSimulator

연합학습의 정상 학습, 내부자 공격·방어, DLG 모델 역추적을 실행하고 결과를 비교·분석하는 시뮬레이터입니다.

## 주요 기능

- CIFAR-10 기반 연합학습과 FedAvg
- 내부자 협력 공격, 노름 클리핑, 좌표별 중앙값 방어
- 참여율·지연·Non-IID 설정을 포함한 연속 연합학습 24개 조합
- 연속 연합학습 결과의 정상 모델 4개를 사용하는 DLG 12개 조합 통합 실행
- 실험 로그, 체크포인트, 데이터 분할 정보를 보존하는 결과 재실행 및 비교
- Flask에 의존하지 않는 독립 학습·공격 비교 모듈 (`fedsec/`)

## 빠른 시작

PowerShell에서 프로젝트 루트 기준으로 실행합니다.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

공식 CIFAR-10 Python 배치를 준비하고 `exp_manager/.env`에서 데이터 경로를 설정합니다.
예시 설정은 [`exp_manager/.env.example`](exp_manager/.env.example)에 있습니다.
데이터 준비, 환경 변수 및 실행 옵션은 [프로젝트 가이드](docs/PROJECT_GUIDE.md)를 참고하세요.

실험 관리자를 시작합니다.

```powershell
.\.venv\Scripts\python.exe exp_manager\app.py
```

웹 UI: [http://127.0.0.1:8000](http://127.0.0.1:8000). 실험을 시작하기 전에 웹 UI에서 데이터 분배를 실행하세요.
관리자 프로세스가 중앙 서버와 클라이언트 프로세스 10개를 관리합니다.

## 서비스 포트 및 결과

| 서비스 | 기본 포트 |
| --- | ---: |
| 실험 관리자 | 8000 |
| 중앙 서버 | 5000 |
| 클라이언트 | 5101–5110 |

결과는 `results/` 아래 실험별 폴더에 저장됩니다. 통합 실험은 `results/<통합실험 ID>/` 아래에
매니페스트와 각 실험 결과를 함께 저장합니다.

## 테스트

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```
