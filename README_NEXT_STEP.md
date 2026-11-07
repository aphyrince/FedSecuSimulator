# FedSecuSimulator — 보안 알고리즘 2단계 시작 파일

이 자료는 **이미 존재하는 security/defenses.py를 수정하지 않습니다.**

1. 기존 FedSecuSimulator 프로젝트 폴더에 `experiments/`, `tests/`, `docs/`의 파일을 복사합니다.
2. 기존 `tests/test_defenses.py`는 그대로 유지합니다.
3. 프로젝트 최상위에서 `python -m pytest tests/test_defenses.py tests/test_attack_defense_demo.py -v` 실행
4. 프로젝트 최상위에서 `python -m experiments.attack_defense_demo` 실행
5. 자동 생성된 `results/attack_defense_demo/summary.csv`, `client_update_log.csv` 확인
6. `docs/WORK_LOG.md`의 버전, 결과와 관찰을 기입합니다.

보충: 사용자의 기존 `security/defenses.py`가 `clip_update`, `coordinate_median` 두 함수를 가진다고 가정합니다.

최초 학습 정확도가 아닌 **모의 업데이트 수치 검증** 단계임에 주의하세요.
