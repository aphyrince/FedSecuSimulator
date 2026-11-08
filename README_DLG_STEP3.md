# FedSecuSimulator 보안 담당: 3단계 DLG 이미지 복원 실습

> 2026-11-08. 이전 `security/defenses.py`, `experiments/attack_defense_demo.py`,
> 기존 테스트·로그는 수정하지 않는 별도 추가 파일이다.

## 1. 목표

- 원본 이미지 한 장으로부터 모델 가중치에 대한 **1회 그래디언트** 생성
- 모델과 관측 그래디언트만 가지고 **임의 이미지와 임의 레이블을 동시 최적화**
- 원본은 **공격이 끝난 뒤 MSE 계산**에만 사용
- `summary.json`, `loss_history.csv`, `original.png`, `reconstructed.png`,
  `comparison.png` 저장
- 실험 속도를 위해 **교육용 한 계층 선형 분류기**를 사용. 최종 CIFAR-10 연합학습 모델의
  정확도나 개인정보 위험을 실험적으로 입증한 결과로 해석하면 안 됨

## 2. 폴더 복사

아래 다섯 개 파일을 Git 프로젝트의 **최상위 폴더를 기준으로** 복사한다.
`security/defenses.py` 같은 기존 파일은 **덮어쓰지 않는다**.

```text
security/dlg.py
experiments/dlg_experiment.py
tests/test_dlg.py
docs/DLG_LOG_TEMPLATE.md
README_DLG_STEP3.md
```

VS Code 터미널에 `(.venv)`가 표시되면 기존 가상환경을 사용한다.
가상환경을 활성화하지 않았다면 `.venv\Scripts\Activate.ps1` 또는
`.\.venv\Scripts\python.exe`로 Python을 실행한다.

## 3. 라이브러리 점검

```powershell
python -c "import torch, PIL; print(torch.__version__)"
python -m pytest -v
```

`PIL`이 없을 경우:

```powershell
python -m pip install pillow
```

실제 CIFAR-10을 불러오려면 별도로 다음 라이브러리가 필요하다.

```powershell
python -m pip install torchvision
```

`torch`와 `torchvision`의 바이너리 버전이 맞지 않아 오류가 발생하면
PyTorch 공식 설치 페이지에서 **같은 환경에 맞는 한 쌍**을 설치한다.

## 4. 먼저 작은 이미지로 실습

```powershell
python -m experiments.dlg_experiment --source synthetic --steps 60
```

출력 위치: `results/dlg/synthetic/`

- `summary.json`: 실제 레이블(평가용), 예측 레이블, MSE, 그래디언트 차이, 시드
- `loss_history.csv`: 각 반복 단계의 그래디언트 일치 손실
- `original.png`: 생성한 실험 이미지
- `reconstructed.png`: 공유된 그래디언트로 추정한 이미지
- `comparison.png`: 두 이미지를 옆으로 붙인 비교용 이미지

`gradient_match_loss`는 공격 알고리즘이 직접 줄이는 손실값이다.
`MSE`는 원본 이미지를 사용해 *실험 종료 후* 계산하는 평가 지표이다.

## 5. 이후 CIFAR-10으로 확장

```powershell
python -m experiments.dlg_experiment --source cifar10 --index 25 --steps 200 --out-dir results/dlg/cifar10_25
```

- `torchvision.datasets.CIFAR10(train=True)`의 25번 이미지 사용
- 첫 실행에는 CIFAR-10 데이터 다운로드가 필요하여 네트워크 연결 필요
- 입력 이미지는 `[1, 3, 32, 32]`, 실험 중 값 범위는 `[0,1]`
- 실험 결과는 `results/dlg/cifar10_25`에 저장됨
- 원본과 정답은 **복원 함수에 전달되지 않고**, 그래디언트 생성·사후 평가에만 사용
- 200회 수행해도 MSE 감소 또는 정확한 복원은 보장되지 않음

## 6. 코드 주요 흐름

1. `TinyDLGNet` : 실험에 사용할 고정 모델 생성
2. `compute_shared_gradients` : 원본 이미지와 라벨로 관측 그래디언트 만들기
3. `reconstruct_from_gradients` : 더미 이미지 및 라벨 분포를 반복 최적화
4. `run` : 원본 이미지·복원 이미지 비교, CSV와 JSON 저장

핵심: `torch.autograd.grad(..., create_graph=True)`로 **그래디언트를 미분할 수 있는
계산 그래프**를 만든다. 모델 가중치는 변경하지 않고, 더미 이미지만 변경한다.

## 7. Git 정리

`data/` 폴더는 CIFAR-10 원본 다운로드 파일로 용량이 매우 크므로 `.gitignore`에
`/data/`를 넣어 업로드를 방지한다. `.venv/`, `.pytest_cache/`도 제외한다.

```powershell
git status
git branch --show-current
git add security/dlg.py experiments/dlg_experiment.py tests/test_dlg.py docs/DLG_LOG_TEMPLATE.md README_DLG_STEP3.md
git commit -m "feat: add single-image DLG reconstruction experiment"
git push -u origin HEAD
```

시험 결과를 확인한 다음 요약·이미지 파일은 필요한 경우에만 별도로 커밋한다.

## 8. 근거 자료와 코드 출처

- Zhu, Liu, Han (NeurIPS 2019), **Deep Leakage from Gradients**
  https://papers.neurips.cc/paper_files/paper/2019/hash/60a6c4002cc7b29142def8871531281a-Abstract.html
- 저자 공개 코드: https://github.com/mit-han-lab/dlg (MIT 라이선스)
- PyTorch autograd.grad: https://docs.pytorch.org/docs/stable/generated/torch.autograd.grad.html

이 실습 파일은 위 아이디어(관측/더미 그래디언트 일치)를 바탕으로
직접 작성한 **교육용 간소화 구현**이다. 원본 공개 구현 코드를 복사해 사용하지 않았다.
