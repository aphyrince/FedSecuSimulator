from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class ExpResult:
    """실험 결과 데이터"""

    experiment_id: str
    """실험 고유 ID"""

    round_results: list[dict[str, Any]] = field(default_factory=list)
    """라운드별 학습 결과 목록
    예: [
        {
            "round": 1,
            "accuracy": 0.85,
            "loss": 0.45
        }
    ]
    """

    dlg_results: list[dict[str, Any]] = field(default_factory=list)
    """DLG 공격 실험 결과 목록
    예: [
        {
            "client_id": 1,
            "reconstruction_loss": 0.12
        }
    ]
    """

    started_at: datetime = field(default_factory=datetime.now)
    """실험 시작 시각"""

    finished_at: datetime | None = None
    """실험 종료 시각 (실험 진행 중에는 None)"""
