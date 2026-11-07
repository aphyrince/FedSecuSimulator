from dataclasses import dataclass
from datetime import datetime
from typing import Literal


@dataclass
class TrainingState:
    """클라이언트의 로컬 학습 상태"""

    round_id: int
    """현재 학습 라운드 번호"""

    model_version: int
    """학습에 사용한 글로벌 모델 버전"""

    status: Literal["pending", "training", "completed", "failed"] = "pending"
    """학습 진행 상태
    - pending: 학습 대기
    - training: 학습 진행 중
    - completed: 학습 완료
    - failed: 학습 실패
    """

    num_samples: int = 0
    """이번 학습에 사용한 데이터 표본 수"""

    local_epochs: int = 1
    """로컬 학습 반복 횟수"""

    batch_size: int = 32
    """학습 배치 크기"""

    learning_rate: float = 0.001
    """학습률"""

    started_at: datetime | None = None
    """학습 시작 시각"""

    finished_at: datetime | None = None
    """학습 종료 시각"""
