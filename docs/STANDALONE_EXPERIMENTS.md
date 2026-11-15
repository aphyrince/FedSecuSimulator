# 독립 연합학습 및 모델 실험 가이드

이 문서는 Flask 서버 없이 CIFAR-10 학습·공격 비교와 E01~E37 모델 실험을 실행하는 방법을 설명합니다.
웹 기반 운영 실험은 [프로젝트 가이드](PROJECT_GUIDE.md)의 실험 모드를 참고하세요.

## 실행 방식

`fedsec` CLI는 한 프로세스에서 클라이언트 데이터셋을 순차 학습합니다. 정상 학습과 내부자 공격 비교는
동일한 데이터 배정·학습 조건·초기 모델의 독립 복사본으로 실행합니다. 이 독립 비교 경로는 부분 참여,
지연 클라이언트 및 방어 실험용이 아니며, 해당 조건은 Flask 실험 관리자에서 설정합니다.

프로젝트 루트에서 설정을 확인하려면 `--dry-run`을 사용합니다.

```powershell
.\.venv\Scripts\python.exe -m fedsec --config configs/fedsec.example.json --dry-run
```

실제 실행 시 설정 파일에는 데이터 매니페스트가 필요합니다. 여러 설정을 한 번의 CLI 호출에 지정할 수
있으며, 기존 결과와 겹치지 않는 실험 ID를 사용합니다.

```powershell
.\.venv\Scripts\python.exe -m fedsec --config configs/fedsec.example.json --experiment-id standalone_check_002
```

`configs/fedsec.example.json`은 연결 확인용 작은 설정입니다. 실제 성능 실험에는 목적에 맞는 데이터·라운드·
학습 설정을 사용하세요. 결과는 기본적으로 `results/standalone/<실험 ID>/`에 저장됩니다.
각 폴더의 `config.json`은 후속 실험의 기준 설정으로 재사용할 수 있습니다.

Python API에서는 `fedsec.ExperimentSettings`, `fedsec.data.load_datasets`,
`fedsec.run_comparison_experiment`를 사용할 수 있습니다. API는 실행 결과를 반환하고, CLI가 JSON 결과와
체크포인트 파일을 저장합니다.

## CIFAR-10 검증 및 최종 테스트 데이터

`configs/cifar10_45k_5k.json`은 CIFAR-10 훈련 배치에서 45,000장을 학습용으로, 5,000장을 검증용으로
나눕니다. 클라이언트 배정과 검증 데이터는 범위 매니페스트로 지정되며, 학습·클라이언트·평가 간 중복은
거부됩니다.

E29~E37은 학습 중 검증 결과와 공식 테스트 결과를 분리합니다. 학습이 끝난 뒤 `test_batch`의 10,000장을
`final_test_ranges`로 추가해 평가합니다. 이 테스트 범위는 학습 및 검증 범위와 겹치지 않아야 합니다.

## E01~E05: 모델 기준 실험

- **E01**: ResNet-18의 1라운드 학습·검증 및 체크포인트 저장 확인
- **E02**: SmallCNN 기준 설정, 50라운드, 로컬 1 epoch, 배치 32, 학습률 `0.001`, SGD 모멘텀 `0`
- **E03~E05**: SmallCNN, 4층 CNN, ResNet-18 비교. 학습률 `0.03`, SGD 모멘텀 `0.9`

E02~E05는 시드 `42`와 동일한 학습·검증 데이터 배정을 사용합니다.

```powershell
.\.venv\Scripts\python.exe -m experiments.run_model_trials --config configs/e01_resnet18.json --dry-run
.\.venv\Scripts\python.exe -m experiments.run_model_trials --config configs/e02_smallcnn_baseline.json --config configs/e03_smallcnn.json --config configs/e04_cnn4.json --config configs/e05_resnet18.json --dry-run
```

설정을 확인한 뒤 실제 학습을 하려면 `--dry-run`을 제거합니다.

## E06~E28: 설정 탐색

| 실험 | 비교 내용 |
| --- | --- |
| E06~E13 | 선택한 두 모델 각각에서 학습률 `0.01`, `0.02`, `0.05`, `0.10` 비교 |
| E14~E17 | 선택한 모델·학습률에서 드롭아웃 `0.1`, `0.2`, `0.3`, `0.4` 비교 |
| E18~E22 | 좌우 반전, 랜덤 크롭, 조합, ColorJitter 및 Cutout 증강 비교 |
| E23~E26 | 라운드×로컬 epoch `30×1`, `65×1`, `80×1`, `40×2` 비교 |
| E27 | 선택한 SGD 설정을 Adam(학습률 `0.001`)으로 재검증 |
| E28 | 최종 후보와 다른 드롭아웃 차순위 후보 재검증 |

예를 들어 E06~E13에서 E03과 E04를 비교하려면 다음처럼 실행합니다. `--trial E06`을 추가하면 해당
단일 실험만 설정·실행할 수 있습니다.

```powershell
.\.venv\Scripts\python.exe -m experiments.run_learning_rate_trials --models E03 E04 --dry-run
```

다음 단계는 이전 단계에서 선택한 설정을 결과 폴더의 `config.json`으로 전달합니다.

```powershell
$python = '.\.venv\Scripts\python.exe'
& $python -m experiments.run_dropout_trials --base-config $chosenModelConfig --learning-rate 0.02 --dry-run
& $python -m experiments.run_augmentation_trials --base-config $chosenDropoutConfig --dry-run
& $python -m experiments.run_training_schedule_trials --base-config $chosenAugmentationConfig --dry-run
& $python -m experiments.run_optimizer_trials --base-config $chosenScheduleConfig --trial E27 --dry-run
& $python -m experiments.run_optimizer_trials --base-config $chosenScheduleConfig --dropout-base-config $chosenOptimizerConfig --runner-up-dropout 0.2 --trial E28 --dry-run
```

`$chosen...Config`은 앞 단계에서 선택한 결과의 설정 파일 경로로 바꿉니다. `0.2`는 E28의 예시이므로
실제 차순위 드롭아웃 값으로 변경합니다. `--dry-run`으로 실제 학습 없이 설정을 확인할 수 있습니다.
`--trial` 옵션은 해당 모듈에서 지원할 때 사용해 개별 실험만 선택하세요.

## E29~E37: 다중 시드 최종 비교

세 시드 `101`, `202`, `303`에서 기준 SmallCNN 정상 학습, 선택 모델 정상 학습, 선택 모델 공격 학습을
비교합니다.

- **E29~E31**: E02 기준 설정의 정상 학습
- **E32~E34**: 선택한 최종 설정의 정상 학습
- **E35~E37**: 같은 최종 설정의 내부자 공격 학습

E32~E37은 선택한 `--base-config`가 E02와 같은 학습·검증 데이터 배정을 사용해야 합니다. 각 라운드는
검증셋으로 평가하고, 학습 종료 후 공식 테스트셋 10,000장으로 별도 평가합니다. `--trial E번호`로 특정
실험만 선택할 수 있습니다.

```powershell
.\.venv\Scripts\python.exe -m experiments.run_final_trials --base-config $chosenFinalConfig --dry-run
```

공격 설정에 악성 클라이언트가 지정되지 않은 경우 기본 클라이언트 ID `0`, `1`을 사용합니다. 명시하는 경우
데이터 배정에 포함된 클라이언트 두 개를 지정해야 합니다.

## 결과 파일

일반 실험은 `results/standalone/<실험 ID>/` 아래에 설정과 각 실행 결과를 저장합니다.

- `config.json`: 실행 설정과 데이터 매니페스트
- `normal/results.json`, `attack/results.json`: 라운드별 평가와 최종 요약
- `normal/final_model.pt`, `attack/final_model.pt`: 실행별 최종 체크포인트
- `comparison.json`: 정상·공격 비교 실행의 정확도 차이
- `normal/rounds/`, `attack/rounds/`: 완료된 라운드 기록

실험이 중단되면 완료된 라운드 기록은 남습니다. 기존 결과 폴더를 덮어쓰지 않도록 새 실험 ID를 사용하세요.

## 학습 코드의 주요 구성

| 경로 | 역할 |
| --- | --- |
| `fedsec/model.py`, `fedsec/models/` | 독립 실험의 모델 구조와 전처리 |
| `fedsec/data.py` | 매니페스트의 CIFAR-10 범위·인덱스 로딩 및 데이터 겹침 검증 |
| `fedsec/training.py` | 로컬 학습 및 실제 분류 정확도·손실 평가 |
| `fedsec/aggregation.py` | FedAvg 집계 |
| `fedsec/attack.py` | 내부자 협력 공격 업데이트 생성 |
| `common/model.py` | 서버·클라이언트 모델, 텐서 및 체크포인트 변환 |
| `clients/services/local_training.py` | 클라이언트 로컬 학습 |
| `server/services/fedavg.py` | 서버 FedAvg 업데이트 집계 |
| `server/services/model_evaluation.py` | 서버 테스트 데이터 평가 |
| `clients/services/collusion.py` | 클라이언트 협력 공격 업데이트 |

독립 모델 탐색에서 ResNet-18 및 데이터 증강을 사용할 때는 `torchvision` 구현을 사용합니다. 반면 매니저 기반
서버·클라이언트 실행은 REST API를 유지하며 프로젝트의 공통 모델 및 업데이트 형식을 사용합니다.
기존 API의 JSON 키를 유지하고, `weights`와 `update.values`에는 파라미터별 중첩 실수 배열을 전달합니다.
