# FedSecuSimulator

연합학습의 정상 학습, 내부자 공격·방어, DLG 모델 역추적을 실행하고 결과를 비교·분석하는 시뮬레이터입니다.

## 주요 기능

- CIFAR-10 기반 연합학습과 FedAvg
- 내부자 협력 공격, 노름 클리핑, 좌표별 중앙값 방어
- 참여율·지연·Non-IID 설정을 포함한 연속 연합학습 24개 조합
- 정상 체크포인트를 사용하는 통합 연속 FL + DLG 실험
- 로그·체크포인트·데이터 분할 정보를 보존하는 결과 비교와 재실행
- Flask 없이 실행하는 독립 학습·공격 비교 모듈 (`fedsec/`)

## 빠른 시작

PowerShell에서 프로젝트 루트 기준으로 가상환경과 의존성을 준비합니다.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

공식 CIFAR-10 Python 배치를 준비하고 `exp_manager/.env`에서 데이터 경로를 설정합니다.
예시는 [`exp_manager/.env.example`](exp_manager/.env.example)에 있습니다.
상세한 데이터 준비와 독립 모델 실험 절차는 [프로젝트 가이드](docs/PROJECT_GUIDE.md)를 참고하세요.

실험 관리자를 시작합니다.

```powershell
.\.venv\Scripts\python.exe exp_manager\app.py
```

웹 UI는 [http://127.0.0.1:8000](http://127.0.0.1:8000)에서 열립니다. 실험 시작 전에 데이터 분배를 실행하세요.
관리자가 중앙 서버와 클라이언트 프로세스 10개를 관리합니다.

## 서비스 및 결과

| 서비스 | 기본 포트 |
| --- | ---: |
| 실험 관리자 | 8000 |
| 중앙 서버 | 5000 |
| 클라이언트 | 5101–5110 |

결과는 `results/` 아래 실험별 폴더에 저장됩니다. 통합 실험 결과는 `results/<통합실험 ID>/` 아래에 모입니다.

## 테스트

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```
