from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class ExpResult:
    # 실험 식별자
    experiment_id: str

    # 라운드별 학습 결과
    round_results: list[dict] = field(default_factory=list)
    # 결과에는 최종 정확도, Loss, MSE 등의 수치와 파일 경로를 저장

    # DLG 공격 결과
    dlg_results: list[dict] = field(default_factory=list)

    # 실험 시작 및 종료 시간
    started_at: datetime | None = None
    finished_at: datetime | None = None
