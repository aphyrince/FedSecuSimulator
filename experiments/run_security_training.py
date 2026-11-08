"""1번 fedsec 독립 실행기에 2번 보안 집계를 연결하는 작은 CLI.

통합 브랜치에서, 프로젝트 루트에서 실행:
 python -m experiments.run_security_training --config configs/development.json \
    --defense norm_clipping --clip-max-norm 1.0 --mode compare --dry-run
"""
from __future__ import annotations

import argparse
import json
from dataclasses import replace

from security.integration import aggregate_with_defense


def main():
    parser = argparse.ArgumentParser(description="1번 FedAvg + 2번 보안 집계 연결 실험")
    parser.add_argument("--config", default="configs/development.json")
    parser.add_argument("--defense", choices=("norm_clipping", "coordinate_median"), required=True)
    parser.add_argument("--clip-max-norm", type=float, required=True,
                        help="정상 업데이트 노름 분포를 참고해 정한 C (>0)")
    parser.add_argument("--mode", choices=("normal", "attack", "compare"), default="compare")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    # 1번의 fedsec 코드는 통합 브랜치에 있어야 한다.
    from fedsec.config import DefenseConfig, load_config
    from fedsec.simulation import execution_plan, simulate

    config = load_config(args.config)
    config = replace(config, defense=DefenseConfig(
        enabled=True, type=args.defense,
        parameters={"max_norm": args.clip_max_norm},
    ))
    if args.dry_run:
        # 실제 학습/데이터 다운로드/파일 저장 없이 설정만 확인.
        # 클리핑 상한도 통합 함수와 동일하게 검사한다.
        from security.integration import _get_clip_bound
        _get_clip_bound(config.defense)
        print(json.dumps(execution_plan(config, args.mode), ensure_ascii=False, indent=2))
        print("설정 검증 완료: 실제 학습은 실행하지 않았습니다.")
        return
    simulate(config, mode=args.mode, run_name=args.run_name,
             aggregation_fn=aggregate_with_defense)


if __name__ == "__main__":
    main()
