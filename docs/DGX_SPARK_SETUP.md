# DGX Spark 실행 안내

이 문서는 NVIDIA DGX Spark를 켠 뒤 FedSecuSimulator를 실행하는 과정을 처음부터 설명합니다.
DGX Spark의 Ubuntu 24.04 ARM64 환경에서는 NVIDIA PyTorch 컨테이너를 사용합니다. Windows PC에서는
프로젝트의 기존 venv 설치 방법을 사용하세요.

## 1. DGX Spark 로그인 및 터미널 열기

1. DGX Spark를 켜고 Ubuntu 초기 설정을 마칩니다. 네트워크에 연결되어 있어야 합니다.
2. 화면이 연결되어 있다면 Ubuntu에 로그인하고 `Ctrl+Alt+T`를 눌러 터미널을 엽니다.
3. 먼저 장비와 Docker가 정상인지 확인합니다.

   ```bash
   whoami
   hostname -I
   nvidia-smi
   sudo docker --version
   ```

`nvidia-smi`에 GPU 정보가 표시되어야 합니다. Docker 명령에서 권한 오류가 나지 않도록 아래 절차에서는
`sudo docker`를 사용합니다. `nvidia-smi`가 없거나 GPU 오류가 나면 프로젝트 설치를 계속하지 말고 DGX OS의
NVIDIA Dashboard에서 시스템 상태와 업데이트를 확인하세요. 임의로 CUDA 드라이버를 설치하지 마세요.

## 2. Git 설치 및 저장소 복제

Git이 이미 설치되어 있으면 첫 번째 줄은 건너뛰어도 됩니다.

```bash
sudo apt update
sudo apt install -y git
mkdir -p ~/projects
cd ~/projects
git clone https://github.com/aphyrince/FedSecuSimulator.git
cd FedSecuSimulator
```

`git clone`이 성공하면 프로젝트 루트에 `Dockerfile.dgx-spark` 파일이 있어야 합니다.

## 3. CIFAR-10 데이터 준비

프로젝트가 사용하는 CIFAR-10 Python 데이터 파일을 다운로드하고 압축을 풉니다.

```bash
mkdir -p data/raw
cd data/raw
wget https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
tar -xzf cifar-10-python.tar.gz
cd ~/projects/FedSecuSimulator
ls data/raw/cifar-10-batches-py
```

마지막 명령에서 `data_batch_1`부터 `data_batch_5`, `test_batch`, `batches.meta`가 보여야 합니다.

## 4. 데이터 경로 설정

예제 환경 파일을 개인 설정 파일로 한 번 복사합니다. `-n` 옵션은 이미 설정 파일이 있으면 덮어쓰지 않습니다.

```bash
cp -n exp_manager/.env.example exp_manager/.env
```

기본 예제는 `SOURCE_DIR=data/raw`입니다. 데이터가 위 안내대로 `data/raw/`에 있다면 그대로 사용하면 됩니다.
다른 위치에 데이터를 두었다면 다음처럼 편집합니다.

```bash
nano exp_manager/.env
```

`SOURCE_DIR`를 프로젝트 루트 기준 상대 경로로 지정합니다. 예를 들어 데이터 폴더가 `source/`라면
`SOURCE_DIR=source`로 바꿉니다. `nano`에서는 `Ctrl+O`, `Enter`로 저장하고 `Ctrl+X`로 나옵니다.

결과를 호스트에 저장할 폴더도 만듭니다.

```bash
mkdir -p results
```

## 5. DGX용 Docker 이미지 만들기

프로젝트 루트(`~/projects/FedSecuSimulator`)에서 실행합니다.

```bash
sudo docker build -f Dockerfile.dgx-spark -t fedsecusim:dgx-spark .
```

첫 빌드는 NVIDIA PyTorch 이미지를 내려받고 프로젝트 의존성을 설치하므로 시간이 걸릴 수 있습니다. NGC 인증
오류가 날 때만 NGC API 키를 준비한 뒤 로그인합니다.

```bash
sudo docker login nvcr.io
```

로그인 프롬프트의 사용자 이름은 `$oauthtoken`, 비밀번호는 NVIDIA NGC 웹사이트에서 만든 API 키입니다.
API 키를 이 프로젝트 파일이나 명령어에 직접 적거나 GitHub에 올리지 마세요. 다시 빌드할 때는 같은 빌드
명령을 실행하면 됩니다.

## 6. 컨테이너에서 GPU 확인

실험을 시작하기 전에 PyTorch가 CUDA GPU를 실제로 인식하는지 확인합니다.

```bash
sudo docker run --rm --gpus all --entrypoint python fedsecusim:dgx-spark \
  -c 'import torch; print("torch:", torch.__version__); print("CUDA:", torch.version.cuda); print("available:", torch.cuda.is_available()); assert torch.cuda.is_available(); print("GPU:", torch.cuda.get_device_name(0))'
```

`available: True`와 GPU 이름이 출력되어야 합니다. 여기서 실패하면 웹 UI를 시작하지 말고 아래 문제 해결
절을 확인하세요.

## 7. 실험 관리자 시작

아래 명령은 프로젝트 루트에서 실행합니다. `--network host`를 사용하므로 관리자·서버·클라이언트 프로세스가
기존 localhost REST 통신과 기본 포트를 그대로 사용합니다. 결과 폴더는 호스트와 공유됩니다.

```bash
sudo docker run --rm --gpus all --ipc=host --network host \
  --user "$(id -u):$(id -g)" \
  -e HOME=/tmp \
  -e MPLCONFIGDIR=/tmp/matplotlib \
  --env-file exp_manager/.env \
  -v "$PWD/data:/workspace/FedSecuSimulator/data:ro" \
  -v "$PWD/source:/workspace/FedSecuSimulator/source:ro" \
  -v "$PWD/results:/workspace/FedSecuSimulator/results" \
  fedsecusim:dgx-spark
```

터미널에 `Experiment manager listening on http://127.0.0.1:8000`이 나오면 실행 중입니다. 이 터미널은 실험이
끝날 때까지 열어 둡니다. 실험 관리자가 서버와 클라이언트 10개를 함께 실행합니다.

## 8. 웹 UI 열기 및 첫 시험 실행

### DGX Spark에 브라우저가 있는 경우

DGX Spark의 브라우저에서 [http://127.0.0.1:8000](http://127.0.0.1:8000)을 엽니다.

### 다른 컴퓨터에서 접속하는 경우

SSH가 DGX Spark에서 활성화되어 있어야 합니다. Spark 터미널의 `whoami`와 `hostname -I` 출력에서 Ubuntu
사용자 이름과 IP 주소를 확인한 뒤, Windows PowerShell에서 다음을 실행합니다.

```powershell
ssh -L 8000:127.0.0.1:8000 <Ubuntu사용자이름>@<DGX_IP주소>
```

SSH 연결을 열어 둔 상태로 Windows 브라우저에서 [http://127.0.0.1:8000](http://127.0.0.1:8000)을 엽니다.
서비스 포트를 외부에 직접 공개하는 대신 이 터널을 사용합니다.

SSH 접속 자체가 안 되면 DGX 터미널에서 상태를 확인합니다.

```bash
sudo systemctl status ssh
```

`Unit ssh.service could not be found`라고 나오면 OpenSSH 서버를 설치하고 시작합니다.

```bash
sudo apt install -y openssh-server
sudo systemctl enable --now ssh
```

방화벽이 활성화되어 있다면 조직의 네트워크 정책에 따라 SSH 접속(기본 TCP 22)을 허용해야 할 수 있습니다.

### 웹 UI에서 첫 라운드 실행

1. 먼저 **데이터 분배**를 실행합니다. 데이터 파일 경로 오류가 나면 3~4단계의 경로와 파일 목록을 확인합니다.
2. 첫 확인용으로 연합학습 라운드 수 `1`, 로컬 epoch `1`, 공격 OFF를 설정합니다.
3. 연산 장치는 **CUDA** 또는 **자동**으로 선택합니다. 기본 선택이 CPU라면 바꾸어야 GPU를 사용합니다.
4. 실험을 시작하고 라운드 결과가 표시되는지 확인합니다. 결과는 프로젝트의 `results/` 폴더에 남습니다.

시험이 성공한 다음 원하는 라운드 수와 공격·방어 설정으로 본 실험을 실행합니다. 정상 학습과 공격·방어를
비교할 때는 각 케이스에서 데이터와 학습 설정을 동일하게 유지하세요.

## 9. 종료 및 다음 실행

실험 관리자 터미널에서 `Ctrl+C`를 누르면 컨테이너가 종료됩니다. `--rm`은 컨테이너만 정리하며, 호스트에
마운트된 `results/` 결과는 삭제하지 않습니다. 다시 실행할 때는 7단계의 명령을 사용합니다.

코드를 업데이트한 뒤에는 실행 중인 컨테이너를 먼저 종료하고 프로젝트 루트에서 다음처럼 갱신합니다.

```bash
git pull --ff-only
sudo docker build -f Dockerfile.dgx-spark -t fedsecusim:dgx-spark .
```

## 문제 해결

- **`docker: command not found`**: DGX OS의 Docker 설치 상태를 확인합니다. 일반 Ubuntu용 CUDA 드라이버나
  컨테이너 런타임을 임의로 설치하지 말고 NVIDIA DGX Spark 문서를 따르세요.
- **Docker 권한 오류**: 명령 앞에 `sudo`가 붙었는지 확인합니다.
- **`nvidia-smi`는 되지만 컨테이너에서 CUDA를 못 찾음**: 6단계 명령에 `--gpus all`이 있는지 확인합니다.
  그래도 실패하면 DGX Spark의 NVIDIA Container Toolkit 상태를 점검합니다.
- **NGC 이미지 내려받기 인증 오류**: 5단계의 `sudo docker login nvcr.io`를 실행합니다.
- **CIFAR 파일을 못 찾음**: `exp_manager/.env`의 `SOURCE_DIR`와 `ls data/raw/cifar-10-batches-py` 결과를
  확인합니다.
- **포트를 이미 사용 중이라고 나옴**: 이전 실험 관리자 터미널에서 `Ctrl+C`를 누릅니다. 기본 포트
  `8000`, `5000`, `5101`~`5110`을 다른 프로세스가 사용 중이지 않아야 합니다.
