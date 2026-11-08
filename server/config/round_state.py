from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal


@dataclass
class RoundState:
    """연합학습 라운드 진행 상태"""

    experiment_id: str
    """소속 실험 고유 ID"""

    round_id: int
    """현재 라운드 번호"""

    model_version: int
    """이번 라운드에서 사용할 글로벌 모델 버전"""

    status: Literal[
        "pending", "checking", "training", "aggregating", "completed", "failed"
    ] = "pending"
    """라운드 진행 상태
    - pending: 시작 대기
    - checking: 클라이언트 참여 확인 중
    - training: 클라이언트 학습 진행 중
    - aggregating: 업데이트 집계 중
    - completed: 라운드 완료
    - failed: 라운드 실패
    """

    available_clients: list[int] = field(default_factory=list)
    """참여 가능하다고 응답한 클라이언트 ID 목록"""

    selected_clients: list[int] = field(default_factory=list)
    """이번 라운드에 최종 선정된 클라이언트 ID 목록"""

    received_clients: list[int] = field(default_factory=list)
    """모델 업데이트를 제출한 클라이언트 ID 목록"""

    excluded_clients: list[int] = field(default_factory=list)
    """제출 실패, 응답 지연 등으로 제외된 클라이언트 ID 목록"""

    started_at: datetime = field(default_factory=datetime.now)
    """라운드 시작 시각"""

    deadline: datetime | None = None
    """모델 업데이트 제출 마감 시각"""
