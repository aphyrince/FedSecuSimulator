# FedSecuSimulator — Security Integration Guide

> **담당:** 2번 / 보안 알고리즘  
> **대상 브랜치:** `feature/security`  
> **문서 목적:** 보안 모듈을 1번(연합학습)·3번(Flask/실험 관리) 코드와 연결하기 위한 함수 규격, 처리 순서, 테스트·로그·실험 기준 정리  
> **기준일:** 2026-11-08  
> **상태:** 보안 모듈 독립 구현·테스트 완료 / 팀 실제 모델·서버 통합 전

## 1. 현재 구현 범위

| 기능 | 소스 파일 | 검증 상태 | 팀 통합 시 남은 작업 |
|---|---|---|---|
| 전체 업데이트 L2 노름 클리핑 | `security/defenses.py` → `clip_update()` | 기본 수치 테스트 완료 | 1번의 실제 클라이언트 업데이트에 적용 |
| 좌표별 중앙값 집계 | `security/defenses.py` → `coordinate_median()` | 홀수·짝수 테스트 완료 | 3번의 서버 집계 선택지로 연결 |
| DLG 이미지·레이블 복원 | `security/dlg.py` → `reconstruct_from_gradients()` | 합성·CIFAR-10 단일 이미지 실험 | 1번의 모델 상태와 연결 |
| 가우시안 그래디언트 노이즈 | `security/gradient_noise.py` → `add_gaussian_noise()` | 노이즈 0/0.01/0.1 비교 실험 | 실험 설정·로그를 3번 관리 기능에 연결 |

- 사용자 Windows 환경: Python 3.13.3 / pytest 9.2.0, **20개 테스트 통과**(2026-11-08 대화에서 실행 결과 확인).
- 이 문서는 현재 대화에서 실제 구현·검증한 파일을 기준으로 작성했다. **다른 팀원의 최신 함수·API 경로는 확인되지 않아 아래의 연결 예시는 합의용 규격이며, 팀 전체 통합이 이미 끝났다는 의미가 아니다.**

## 2. 시스템 처리 흐름 및 담당 경계

### A. 내부자 협력 공격 / 전역 모델 정확도 실험

```text
[3번: 실험 관리자]
  실험 설정·시드·모델 버전 고정 → 해당 라운드 참여자 선택
                  ↓
[1번: 클라이언트 학습]
  동일한 시작 전역 모델로 각 클라이언트 로컬 학습
  update_i = local_parameters_i - global_parameters
                  ↓
[1번: -5배 협력 공격]
  같은 라운드에 참여한 악성 클라이언트(0/1/2명)의 정상 업데이트 평균 × (-5)
  → 참여 악성 클라이언트는 해당 동일 벡터를 제출
                  ↓
[3번: 제출 검증 / 기한 관리]
  round/model_version/client_id·중복·shape·유한값·지연 여부 검증
  → 실제 채택 업데이트 목록 확정 (채택 3명 미만이면 모델 갱신하지 않음)
                  ↓
[2번: 서버측 방어·집계 함수]
  fedavg            : 원본 업데이트 → 표본 수 가중평균
  clipping_fedavg   : 개별 전체 노름 클리핑 → 표본 수 가중평균
  clipping_median   : 개별 전체 노름 클리핑 → 좌표별 중앙값
                  ↓
[3번: 전역 모델 반영·평가]
  next_global_parameters = current_global_parameters + aggregated_update
  검증 정확도/최종 테스트 정확도/채택 수/악성 비율/전후 노름·로그 저장
```

**중요:** `clip_update()`와 `coordinate_median()` 자체에는 악성 클라이언트의 정답 ID를 넘기지 않는다. 악성 ID는 공격 생성·실험 분석에만 사용한다.

### B. DLG 이미지 유출 실험 — FedAvg 성능 실험과 **별도 실행**

```text
[1번 측 모델·데이터] 같은 모델 버전에서 CIFAR-10 이미지 1장과 정답 사용
  → compute_shared_gradients(model, image, label)
  → (선택) add_gaussian_noise(grads, std, seed)  # 전송 전에 적용한다고 가정
  → [서버/공격자가 관측 가능하다고 가정한 그래디언트]
  → reconstruct_from_gradients(model, observed_grads, image_shape, ...)
  → [3번 측 평가/기록] 원본·복원 이미지 비교 MSE, 레이블 일치, 설정/시드/이미지 ID 저장
```

- **학습용 이미지 원본과 정답 레이블은 공격 함수에 주지 않는다.** `MSE` 계산은 복원이 끝난 후 평가 단계에서만 한다.
- DLG에서 사용하는 **단일 이미지의 1회 그래디언트**와, 일반 FedAvg에서 수집하는 **로컬 학습 후 모델 가중치 차이**는 서로 다르다. 둘을 혼용하지 않는다.
- DLG 실험에서 공격자는 같은 모델 **구조 및 그 시점의 가중치**를 알고 있다고 가정한다.
- 현재 제공된 `TinyDLGNet`은 교육용 `Flatten + Linear` 분류기이다. 팀의 실제 CNN으로 전환한 DLG 복원 성능은 아직 검증되지 않았다.

## 3. 팀 간 데이터 계약 — 우선 합의할 내용

| 필드/규칙 | 제안 규격 | 담당 또는 비고 |
|---|---|---|
| `experiment_id` | 문자열 | 3번 생성/기록 |
| `round_id` | 0 또는 1부터 시작 중 팀 내 통일, 정수 | 3번 관리 |
| `model_version` | 시작 전역 모델의 버전/해시 | 3번 검증 |
| `client_id` | 예: 0~9의 정수, 중복 금지 | 1·3번 |
| `sample_count` | 1 이상의 정수, 채택 시 유효성 확인 | 1번 전달/3번 검증 |
| `update` | `list[torch.Tensor]` (동일 파라미터 순서·shape·device·dtype) | 1번 산출 → 2번 처리 |
| 업데이트 값 | **`local - global`**, 가중치 자체 아님 | 1번 |
| 채택 업데이트 | 선택·제출·지연/중복/형식 검증 후 확정된 목록 | 3번 |
| `defense` | `fedavg`, `clipping_fedavg`, `clipping_median` | 3번 실험 설정 |
| `clip_max_norm` | 정상 업데이트 분포를 바탕으로 선정, 동일 비교 실험에서 고정 | 2번 제안·팀 합의 |
| `attack_multiplier` | `-5.0` | 1번 |
| `malicious_ids` | 전체 10개 중 2개 지정; 방어 함수에는 전달 금지 | 1·3번 관리 |
| `seed` | 초기 모델/분할/참여/지연/복원에서 각각 기록 | 3번 |

**텐서 포맷 주의:** PyTorch `torch.Tensor`는 일반 JSON에서 직접 직렬화되지 않는다. 3번의 Flask API가 전송/역직렬화할 자료형(파일·배열·바이너리, dtype/shape/순서/버전)을 팀에서 합의한다. 이 문서는 특정 REST 경로/메시지 포맷이 이미 구현되었다고 가정하지 않는다.

**모델 파라미터 주의:** 신경망의 `named_parameters()`와 `state_dict()`는 동일하지 않을 수 있다(예: BatchNorm 버퍼). 파라미터 이름과 순서를 고정해 매핑하고, 버퍼/비학습 항목은 별도 정책을 정한다. 집계 도중 모델 파라미터 구조가 바뀌면 안 된다.

## 4. 실제 함수 API (현재 구현 코드 기준)

### 4.1 `clip_update(update, max_norm)`

```python
from security.defenses import clip_update

clipped = clip_update(update, max_norm=0.7)
```

- **입력:** `update: list[Tensor]` — *클라이언트 한 명*의 전체 모델 업데이트; `max_norm: float > 0`.
- **반환:** 새로운 `list[Tensor]` (원본 텐서 수정 안 함).
- **계산:** 전체 텐서 원소에 대해 L2 노름을 구한 뒤 `min(1, max_norm / norm)`을 모든 텐서에 동일하게 곱함.
- **현재 검증:** 기본 크기 제한·상한 이내 변경 없음.
- **주의:** 예제의 `0.7`은 2차원 모의실험의 임시 상한이지 실제 CNN에서 확정된 상한이 아님. 통합 전 비유한값/자료형/디바이스/모델 순서 검증은 추가 필요.

### 4.2 `coordinate_median(client_updates)`

```python
from security.defenses import coordinate_median

aggregated_update = coordinate_median(clipped_updates)
```

- **입력:** `client_updates: list[list[Tensor]]` — 채택된 **여러 클라이언트**의 동일 구조 업데이트.
- **반환:** `list[Tensor]` — 파라미터 좌표마다 중앙값이 들어간 집계 업데이트.
- **홀수:** 가운데 값. **짝수:** 가운데 두 값의 평균.
- **가중치:** `sample_count`를 사용하지 않는 **비가중** 좌표별 중앙값.
- **주의:** `clip_update`를 먼저 적용하는지는 호출자(3번의 서버 방어 선택기)가 결정한다. 현재 모의실험의 `clipping_median`은 **클리핑 후 중앙값** 방식이다.

### 4.3 `compute_shared_gradients(model, image, label)`

```python
from security.dlg import compute_shared_gradients

gradients = compute_shared_gradients(model, image, label)
```

- **입력:** PyTorch `nn.Module`, 배치 크기 1의 RGB 이미지 `Tensor`(`[1, 3, H, W]`), 정수 레이블.
- **반환:** 모델 `parameters()` 순서와 대응하는 `tuple[Tensor, ...]`.
- **실행 위치:** DLG 실험에서 원본 데이터를 가진 클라이언트를 재현하는 부분.
- **주의:** 모델을 이 함수 호출 전에 해당 라운드/실험의 동일 가중치로 설정하고, 입력의 전처리·정답 인덱스 정의를 고정해야 한다.

### 4.4 `add_gaussian_noise(gradients, std, seed=42)`

```python
from security.gradient_noise import add_gaussian_noise

shared_gradients = add_gaussian_noise(gradients, std=0.01, seed=42)
```

- **입력:** `tuple[Tensor, ...]`, `std >= 0` 유한 실수, 난수 시드.
- **반환:** 원본 텐서를 건드리지 않는 `tuple[Tensor, ...]`.
- **계산:** 각 좌표에 독립적인 정규 잡음 `N(0, std²)` 추가. `std=0`이면 복사본 반환.
- **주의:** 단순 노이즈 추가는 **정식 차등 개인정보보호(DP) 보장**을 의미하지 않는다. 노이즈가 모델 품질에 미치는 영향은 별도 평가 필요.

### 4.5 `reconstruct_from_gradients(...)`

```python
from security.dlg import reconstruct_from_gradients

result = reconstruct_from_gradients(
    model=model,
    observed_gradients=shared_gradients,
    image_shape=(3, 32, 32),
    num_classes=10,
    steps=200,
    lr=0.1,
    seed=43,
)
reconstructed_image = result.image
predicted_label = result.predicted_label
```

- **입력:** 공격자가 알고 있다고 가정한 모델과 관측 그래디언트, 이미지 크기, 반복 횟수, 학습률, 초기 시드.
- **반환 `DLGResult`:** `image`, `predicted_label`, `match_loss_before`, `match_loss_after`, `history`.
- **최적화 대상:** 임의 이미지와 임의 레이블 로짓. **모델 가중치는 수정하지 않음.**
- **주의:** 공격 함수에 원본 `image`, 실제 `label`은 주지 않으며, MSE 역시 최적화 손실로 사용하지 않는다.

## 5. 클리핑·중앙값 집계 연결 예제

아래 예시는 **현재 코드에서 실제로 실행할 수 있는 모의 업데이트 기반 연결 확인용**이다. 최종 서버에 붙일 때는 `local_fedavg`를 1번/3번의 실제 집계 구현으로 교체한다.

```python
from security.defenses import clip_update, coordinate_median
from experiments.attack_defense_demo import make_updates, local_fedavg

clean, attacked, sample_counts = make_updates()
accepted_updates = attacked      # 예제 전용: 실제 서버에서는 검증·채택된 업데이트만 사용
clip_max_norm = 0.7              # 예제 전용 값

# A. 기존 FedAvg
fedavg_result = local_fedavg(accepted_updates, sample_counts)

# B. 클리핑 후 FedAvg
clipped = [clip_update(u, clip_max_norm) for u in accepted_updates]
clipped_fedavg_result = local_fedavg(clipped, sample_counts)

# C. 클리핑 후 좌표별 중앙값
clipped_median_result = coordinate_median(clipped)

print('FedAvg:', fedavg_result[0])
print('Clipping + FedAvg:', clipped_fedavg_result[0])
print('Clipping + Median:', clipped_median_result[0])
```

**실제 학습 업데이트로 바꾸는 방법(개념 예시):**

```python
# 1번 팀원 구현에 맞추어 정확한 모델·파라미터 인터페이스 조정 필요
# 동일한 파라미터 이름/순서의 전역 모델과 로컬 모델만 비교한다.
update = [
    local_param.detach() - global_param.detach()
    for global_param, local_param in zip(global_model.parameters(), local_model.parameters())
]

# 3번 서버에서 validated_updates로 집계 후 전역 모델에 '델타'를 더한다.
# 이미 집계된 업데이트를 모델 전체 가중치로 덮어쓰면 안 된다.
with torch.no_grad():
    for parameter, delta in zip(global_model.parameters(), aggregated_update):
        parameter.add_(delta)
```

> 위 블록은 **통합 설계 예시**이며, 두 모델의 파라미터 이름·개수·shape가 일치하는지 검사하는 코드와 `torch` import는 생략했다. 실제 프로그램에 그대로 복사하기 전 검증을 추가해야 한다. BatchNorm 등 버퍼 처리도 별도 정책이 필요하다.

## 6. DLG와 노이즈 비교 연결 예제

```python
import torch
from security.dlg import compute_shared_gradients, reconstruct_from_gradients
from security.gradient_noise import add_gaussian_noise

# 다음 변수는 1번 데이터·모델 기능 또는 DLG 실험에서 제공해야 함:
# model: 동일 모델 버전으로 고정된 nn.Module
# image: [1,3,32,32] 형태 이미지 1장 / true_label: 정수 레이블
clean_gradients = compute_shared_gradients(model, image, true_label)

for noise_std in (0.0, 0.01, 0.1):
    visible_gradients = add_gaussian_noise(clean_gradients, std=noise_std, seed=44)
    result = reconstruct_from_gradients(
        model=model, observed_gradients=visible_gradients,
        image_shape=(3, 32, 32), num_classes=10, steps=200, seed=43,
    )
    # 반드시 복원 후 평가 단계에서만 원본을 사용
    mse = torch.mean((result.image - image).square()).item()
    print(noise_std, mse, result.predicted_label)
```

- 동일 이미지·모델 상태·복원 시드·반복 횟수를 고정하고, **비교하는 요인은 `noise_std`만 변경**한다.
- 실제 실험에서는 이미지를 여러 장 사용하고 실험 시드도 바꿔 반복한다. `noise_std=0.01, 0.1`은 현재 초기 실험값이며 확정된 보안 기준이 아니다.
- **원본/실제 정답은 평가자만 접근**하도록 구분한다. DLG 실험용 원본 이미지 파일을 일반 운영 서버에 공개 저장하지 않는다.

## 7. 저장 파일, 실행 방법 및 재현 조건

### 7.1 설치·테스트

프로젝트 최상위(`FedSecuSimulator/`)에서 가상환경 활성화 후 실행한다.

```powershell
python --version
python -m pip show torch torchvision Pillow pytest
python -m pytest -v
```

- 실제 사용 패키지: `torch`, `torchvision`(CIFAR-10 로딩), `Pillow`(PNG 저장), `pytest`(자동 테스트).
- 팀에서 `requirements.txt` 또는 다른 의존성 관리 방식과 정확한 버전을 확정한다. 현재 문서가 다른 팀원 환경에 설치를 완료해 주지는 않는다.
- 2026-11-08 사용자 Windows 출력 기준 **20개 테스트 통과**. 외부 팀 코드 포함 통합 테스트 결과가 아니다.

### 7.2 실험 실행

```powershell
# 가상 10개 클라이언트: FedAvg/클리핑/중앙값 비교
python -m experiments.attack_defense_demo

# 단일 이미지 DLG
python -m experiments.dlg_experiment --source cifar10 --index 25 --steps 200 --out-dir results/dlg/cifar10_25

# 동일 이미지의 노이즈 세 조건 비교
python -m experiments.dlg_noise_experiment --source cifar10 --index 25 --steps 200 --out-dir results/dlg_noise/cifar10_25
```

- CIFAR-10 첫 실행 시 데이터 다운로드가 필요할 수 있다.
- **이미지별로 `--out-dir` 경로를 달리 지정해야 결과 파일 덮어쓰기를 피할 수 있다.**

### 7.3 결과 파일

| 경로 예시 | 주요 자료 |
|---|---|
| `results/attack_defense_demo/summary.csv` | 정상/공격 × 집계 3종의 결과 벡터와 기준 거리 |
| `results/attack_defense_demo/client_update_log.csv` | 클라이언트별 공격/클리핑 전후 노름 |
| `results/dlg/cifar10_25/` | 원본·복원·비교 PNG, `loss_history.csv`, `summary.json` |
| `results/dlg_noise/cifar10_25/` | 노이즈별 복원 PNG/손실 CSV, `summary.csv`, `summary.json`, `comparison.png` |

### 7.4 현재 확보된 단일 이미지 결과 (보고서용 *예시*, 일반화 금지)

CIFAR-10 학습 데이터 `index=25`, 단일 32×32 RGB 이미지, DLG 200회, 교육용 선형 모델:

| 노이즈 표준편차 | MSE | 최종 그래디언트 일치 손실 | 추정 레이블 |
|---:|---:|---:|---:|
| 0 | 0.002257 | 0.281322 | 6 |
| 0.01 | 0.002301 | 3.046037 | 6 |
| 0.1 | 0.011414 | 276.589447 | 6 |

- 정답 레이블(평가 전용) 6, 세 조건 모두 추정 레이블 6. 노이즈 0.1의 MSE는 노이즈 0보다 약 **5.06배** 큼.
- 그 자체로 일반적인 개인정보 보호 성공이나 DP 보장을 뜻하지 않음.

## 8. 팀 통합 시 서버/클라이언트 검증 규칙

1. 모든 클라이언트는 **동일 라운드의 동일 시작 전역 모델**에서 학습한다.
2. 3번은 모델 버전, 라운드, `client_id` 중복, 표본 수, 텐서 수·shape·dtype·장치, NaN/Inf 등을 검증한다.
3. 1번은 같은 라운드 **참여 악성 클라이언트 0/1/2명**에 맞춰 -5배 공격을 생성한다. 실제 서버 채택 여부와 참여 여부는 별도로 기록한다.
4. 지연 클라이언트는 가상 마감 시각 기준으로 제외하고, 실제 PC 순차 실행 시간과 혼동하지 않는다.
5. **채택 업데이트 3개 미만이면 전역 모델을 갱신하지 않는다.** 제외 사유와 채택 수를 기록한다.
6. 3번이 방어 방식 선택 후 **딱 한 번** 집계하고, 집계한 업데이트를 전역 모델에 더한다. `median` 집계 후 `FedAvg`를 또 적용하지 않는다.
7. 방어 함수에 악성 ID를 전달하지 않는다. 라운드·클라이언트·노름(조작 전후 및 방어 전후)·선택/채택 여부·실행/가상 시간 등을 로그에 남긴다.
8. 정확도·공격 피해 등은 **같은 환경·시드·초기 모델·데이터 분할·참여/지연 일정**에서 비교한다. 테스트셋은 최종 평가에만 사용한다.

## 9. 로그 및 성능지표 (3번 담당과 합의 필요)

**학습 실험 최소 로그 필드 제안:** `experiment_id`, `seed`, `round_id`, `client_id`, `model_version`, `attack_mode`, `defense`, `sample_count`, `selected`, `accepted`, `exclusion_reason`, `is_designated_malicious`, `is_actually_attacking`, `norm_before_attack`, `norm_after_attack`, `norm_after_defense`, `virtual_delay`, `deadline`, `wall_time`.

**DLG 실험 최소 로그 필드 제안:** `experiment_id`, `image_id`, `model_version`, `source`, `noise_std`, `steps`, `seed`, `predicted_label`, `true_label_for_evaluation_only`, `mse_for_evaluation_only`, `gradient_match_loss_before`, `gradient_match_loss_after`, `original_image_path`(평가 접근 제한), `reconstructed_image_path`.

**평가 지표(과제 명세 기준):**

- 공격 피해(%p) = 동일 조건의 **정상 FedAvg 정확도 − 공격 FedAvg 정확도**
- 방어 개선(%p) = **공격+방어 정확도 − 공격 FedAvg 정확도**
- 정상 방어 손실(%p) = **정상 FedAvg 정확도 − 정상+방어 정확도**
- 시드별 결과와 평균을 함께 기록; 음수 차이도 수정하지 않고 보고.
- 전체 지정 악성 비율(`2/10`)과 **실제 채택 악성 비율**은 구분. 채택자 수 0이면 N/A.
- DLG의 MSE는 이미지 픽셀 복원 오차, 학습 정확도와 동일한 지표가 아님.

## 10. 통합 직전 체크리스트

- [ ] 1번과 `local-global` 업데이트 정의, 파라미터 이름·순서·shape, 모델 버전 합의
- [ ] 1번과 악성 참여 0/1/2명 공격 생성 테스트 공유
- [ ] 3번과 방어 선택지·함수 호출 시점·채택 3명 미만 처리 합의
- [ ] 3번과 Flask 전송 직렬화 형식 및 제외/오류 응답 합의
- [ ] 실제 모델·클라이언트 업데이트로 **1라운드** 통합 테스트
- [ ] 정상·공격 각각의 FedAvg/클리핑/중앙값 정확도 비교
- [ ] 부분 참여(5/10), 지연 3명, 레이블 편향, 여러 시드 실험
- [ ] 1번 모델의 동일 가중치로 DLG 단일 이미지 복원 검증
- [ ] DLG 노이즈 0/작음/큼 반복 실험 및 시드별 MSE 저장
- [ ] 보고서 원인 분석 3건(공격·방어/운영환경/DLG)과 상황별 권고 2건 작성

## 11. 팀원 간 확인 요청 (그대로 공유 가능)

**1번에게 요청**
- 모델 정의 위치와 초기 가중치 저장/불러오기 방법은?
- 클라이언트별 업데이트는 `local-global`인가? 파라미터 `list[Tensor]` 순서를 공유할 수 있는가?
- 클라이언트별 `sample_count`와 공격 생성 함수는 어디서 반환하는가?
- DLG에 사용할 **같은 모델 버전**의 상태를 어떻게 전달할 것인가?

**3번에게 요청**
- Flask API/실험 관리자에서 업데이트를 받고 검증하는 함수는 어디인가?
- 채택된 업데이트와 표본 수를 방어 집계 함수에 넘기는 위치는 어디인가?
- 방어 모드 선택(`fedavg`, `clipping_fedavg`, `clipping_median`)과 `clip_max_norm` 설정은 어디서 관리하는가?
- 실험 로그와 복원 결과 파일 경로/시드/모델 버전을 어떤 형식으로 저장하는가?

---

### 문서 관리 및 Git 업로드

이 문서를 기존 `FedSecuSimulator/docs/SECURITY_INTEGRATION.md`에 넣고, 보안 담당 브랜치에서 다음을 실행한다.

```powershell
git branch --show-current
git status
git add docs/SECURITY_INTEGRATION.md
git commit -m "docs: add security integration guide"
git push -u origin HEAD
```

> **브랜치 주의:** `feature/security`에서 문서만 커밋한다. 다른 팀의 브랜치를 임의로 병합하거나 파일을 덮어쓰지 않는다. 실제 함수명·API가 합의되면 이 문서의 ‘제안 규격’ 부분을 업데이트한다.
