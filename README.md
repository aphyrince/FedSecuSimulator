# FedSecuSimulator

연합학습의 정상 학습, 내부자 공격·방어, DLG 모델 역추적을 실행하고 결과를 비교·분석하는 시뮬레이터입니다.

## 주요 기능

- CIFAR-10 기반 연합학습과 FedAvg
- 내부자 협력 공격, 노름 클리핑, 좌표별 중앙값 방어
- 참여율·지연·Non-IID 설정을 포함한 연속 연합학습 24개 조합
- 정상 체크포인트를 사용하는 통합 연속 FL + DLG 실험
- 로그·체크포인트·데이터 분할 정보를 보존하는 결과 비교와 재실행
- Flask 없이 실행하는 독립 학습·공격 비교 모듈 (`fedsec/`)

## Windows 빠른 시작

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

## DGX Spark 실행 (Ubuntu 24.04 ARM64)

DGX Spark에서는 NVIDIA가 DGX Spark용 PyTorch playbook에서 사용하는
`nvcr.io/nvidia/pytorch:25.11-py3` 컨테이너를 사용합니다. 이 이미지에는 CUDA 빌드 PyTorch가 포함되어
있으므로 Windows 설치용 `requirements.txt`를 컨테이너에 설치하지 않습니다.
처음 설치하는 경우 [DGX Spark 단계별 실행 안내](docs/DGX_SPARK_SETUP.md)를 참고하세요.

먼저 예제 환경 파일을 복사하고 프로젝트 루트에서 Docker 이미지를 빌드합니다.

```bash
cp exp_manager/.env.example exp_manager/.env
docker build -f Dockerfile.dgx-spark -t fedsecusim:dgx-spark .
```

`.env`가 아직 없다면 `exp_manager/.env.example`을 복사하고 `SOURCE_DIR`를 프로젝트 루트 기준 상대 경로로
설정합니다. CIFAR-10 데이터는 `source/` 또는 `data/` 아래에 두고, 결과는 호스트의 `results/`에 보존합니다.

```bash
docker run --rm --gpus all --ipc=host --network host \
  --env-file exp_manager/.env \
  -v "$PWD/source:/workspace/FedSecuSimulator/source:ro" \
  -v "$PWD/data:/workspace/FedSecuSimulator/data:ro" \
  -v "$PWD/results:/workspace/FedSecuSimulator/results" \
  fedsecusim:dgx-spark
```

로컬에서 웹 UI를 열 때는 `http://127.0.0.1:8000`을 사용합니다. 다른 컴퓨터에서 접속할 때는 SSH 포트 포워딩
(`ssh -L 8000:127.0.0.1:8000 <사용자>@<DGX주소>`)을 사용하면 서비스의 내부 REST 주소를 외부에 노출하지
않아도 됩니다. UI의 연산 장치는 `CUDA` 또는 `자동`으로 선택해야 GPU를 사용합니다.

컨테이너의 GPU 인식 여부는 다음 명령으로 확인할 수 있습니다.

```bash
docker run --rm --gpus all --entrypoint python fedsecusim:dgx-spark \
  -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

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
