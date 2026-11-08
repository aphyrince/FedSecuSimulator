# 1번 + 2번 연합학습 보안 집계 연결 방법

> 적용 대상: `aphyrince/FedSecuSimulator` 1번 `feature/federated-learning`, 2번 `feature/security`  
> 확인 당시 1번 API: `simulate(..., aggregation_fn=...)`, `ClientUpdate.delta: dict[str, Tensor]`, `DefenseConfig(type='norm_clipping'|'coordinate_median', parameters={...})`.  
> 실제 팀 GitHub 파일을 **수정하지 않고** 로컬에서 준비한 연결 코드입니다. 먼저 팀원과 통합 브랜치 운영을 합의하세요.

## 제공 파일 (기존 파일 덮어쓰기 없음)

```text
security/integration.py                  # 집계 연결 함수
experiments/run_security_training.py     # 1번 시뮬레이터 실행 도우미
 tests/test_security_integration.py       # 실제 1번 타입과 통합 검사 6개
 docs/ROLE1_SECURITY_BRIDGE_STEPS.md      # 이 문서
```

기존 `security/defenses.py`, `security/dlg.py`, `security/gradient_noise.py`는 그대로 둡니다. DLG는 연합학습 집계와 분리된 실험이므로 이번 연결 단계에서 변경하지 않습니다.

## 단계 A: 네 feature/security 브랜치에 파일 추가

VS Code에서 **FedSecuSimulator 루트 폴더**를 연 상태로 PowerShell 터미널에서:

```powershell
git branch --show-current
git status
```

`feature/security`인지 확인합니다. 다른 브랜치라면 작업물이 저장된 상태인지 확인한 후:

```powershell
git switch feature/security
```

압축파일의 `security`, `experiments`, `tests`, `docs` 폴더에 있는 **새 파일 4개**를 같은 이름의 프로젝트 폴더에 복사합니다. 기존 폴더 전체를 덮어쓰면 안 됩니다.

```powershell
python -m pytest -q
```

아직 1번의 `fedsec/` 폴더가 없으므로 새 통합 테스트 모듈은 **1개 skip**되는 게 정상입니다. 기존 20개 테스트는 계속 통과해야 합니다. 문제가 없으면:

```powershell
git add security/integration.py experiments/run_security_training.py tests/test_security_integration.py docs/ROLE1_SECURITY_BRIDGE_STEPS.md
git commit -m "feat: prepare role1 security aggregation bridge"
git push origin feature/security
```

## 단계 B: 통합 브랜치에서 1번 코드와 함께 검사 (팀원 합의 후)

**먼저 `git status`로 미커밋 변경사항이 없는지 확인하세요.**

```powershell
git fetch origin
git switch -c integration/role1-security origin/feature/federated-learning
```

이 명령은 **1번의 최신 원격 브랜치를 기반으로 새 로컬 브랜치를 생성**하며 기존 `feature/security` 브랜치를 덮어쓰지 않습니다.

이제 2번 브랜치에서 필요한 소스·테스트·문서만 가져옵니다. `git merge` 대신 파일만 가져오므로 README/requirements 충돌을 피할 수 있습니다.

```powershell
git restore --source origin/feature/security -- security/ experiments/ tests/test_defenses.py tests/test_attack_defense_demo.py tests/test_dlg.py tests/test_gradient_noise.py tests/test_security_integration.py docs/SECURITY_INTEGRATION.md docs/ROLE1_SECURITY_BRIDGE_STEPS.md
```

가져온 파일 목록을 확인합니다.

```powershell
git status
python -m pytest -q
```

이제는 1번의 `fedsec/`가 있으므로 **새 통합 테스트 6개가 skip 없이 실행되어야 합니다.** 기존 20개와 합치면 2번 테스트 26개가 되고, 1번의 테스트들이 추가로 실행됩니다. 테스트 총개수는 1번 브랜치 변경에 따라 달라질 수 있습니다.

> 1번의 requirements에는 특정 버전의 torch/torchvision이 적혀 있습니다. 무작정 기존 가상환경을 재설치하거나 강제로 다운그레이드하지 말고, 먼저 현재 가상환경에서 테스트한 후 팀원과 버전을 통일하세요.

성공하면 파일을 통합 브랜치에 **로컬 커밋**합니다.

```powershell
git add security experiments tests docs/SECURITY_INTEGRATION.md docs/ROLE1_SECURITY_BRIDGE_STEPS.md
git commit -m "feat: integrate federated learning with security aggregation"
```

팀에서 합의한 뒤에만 이 브랜치를 원격에 공유하세요.

```powershell
git push -u origin integration/role1-security
```

## 단계 C: 다운로드·학습 없이 연결 설정만 확인

```powershell
python -m experiments.run_security_training --config configs/development.json --defense norm_clipping --clip-max-norm 1.0 --mode compare --dry-run
```

`--dry-run`은 학습을 진행하지 않습니다. 위 `1.0`은 **설정 확인용 예시**이며 실제 모델에 사용할 최종 클리핑 상한 C는 아닙니다.

## 단계 D: 실제 1라운드 미니 실험

1번이 만든 `configs/development.json`은 1라운드·클라이언트당 작은 표본 수를 이용합니다. 모델 학습이 시작되기 전에 CIFAR-10 원본 데이터가 1번 코드가 참조하는 `data/raw` 경로에 있어야 합니다.

```powershell
python -m fedsec download-data --config configs/development.json
python -m fedsec simulate --config configs/development.json --mode normal --run-name baseline_dev_001
```

정상 업데이트 노름을 확인합니다.

```powershell
python -c "from pathlib import Path; from fedsec.artifacts import decode_update; from fedsec.tensors import state_norm; p=Path('outputs/baseline_dev_001/normal/round_0001/submitted_updates'); print([(f.name, round(state_norm(decode_update(f.read_bytes()).delta), 6)) for f in sorted(p.glob('*.pt'))])"
```

위 분포를 확인한 다음 **팀원과 비교 실험 전반에 동일하게 사용할 클리핑 C를 합의**합니다. 아래는 C를 1.0이라고 **가정한 실행 문법 예시**입니다. 실제 값에 맞게 바꾸세요.

```powershell
python -m experiments.run_security_training --config configs/development.json --defense norm_clipping --clip-max-norm 1.0 --mode compare --run-name clipping_dev_001
python -m experiments.run_security_training --config configs/development.json --defense coordinate_median --clip-max-norm 1.0 --mode compare --run-name median_dev_001
```

`outputs/<run-name>/`에서 정상·공격의 `metrics.csv`, `summary.json`, `comparison.json`을 확인합니다. 같은 `run-name`을 재사용하면 1번 실행기는 덮어쓰지 않고 오류를 냅니다.

## 동작 설명

- 1번은 `ClientUpdate.delta`를 **파라미터 이름 → Tensor** 딕셔너리로 전달합니다.
- `security.integration.aggregate_with_defense()`는 이 값을 2번의 **리스트 기반** `clip_update()`에 넘깁니다.
- `norm_clipping`: 클리핑한 델타를 원래 `ClientUpdate`로 만들어 1번의 표본 수 가중 `fedavg()`에 넘깁니다.
- `coordinate_median`: 클리핑 후 좌표별 중앙값을 구하고 전역 모델에 델타를 **한 번** 더합니다. 이 경로에 FedAvg를 또 적용하지 않습니다.
- 공격자의 ID는 집계 함수에 전달되지 않습니다.
- 채택 업데이트가 3개 미만이면 연결 함수는 원래 모델 가중치의 복사본을 반환합니다. **주의:** 1번의 현재 `simulation.py`는 호출 후 모델 버전을 증가시키므로, 서버 운영 단계에서 '모델 갱신 생략' 시 버전/로그 처리 규칙을 3번과 별도로 맞춰야 합니다.
- 이 파일은 **연합학습 방어 연결만** 담당합니다. DLG/노이즈 공격·방어는 별도의 단일 이미지 그래디언트 실험이며 이번 연결 함수에서 사용하지 않습니다.

## 1번과 별도로 합의할 사항

1. 부분 참여에서 악성 클라이언트가 **1명만 참여하는 경우**: 과제 명세의 -5배 단독 공격과 1번 기본 설정의 `min_colluders=2, skip_attack`이 다릅니다.
2. 실제 클리핑 상한 C: 정상 업데이트 노름 분포를 통해 결정하고 비교 조건에서 고정합니다.
3. `fedsec.ClientUpdate`를 공통 자료형으로 삼고 3번 서버의 자체 `ClientUpdate`와 중복되지 않게 합니다.
4. 팀 공통 `torch`, `torchvision`, `Pillow`, `pytest` 버전을 합의합니다.
5. 실제 3번 서버에서는 유효 업데이트 3개 미만일 때 **모델·버전·라운드 로그**를 일관되게 처리합니다.
