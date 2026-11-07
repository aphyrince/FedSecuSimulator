from dataclasses import dataclass, field
from typing import Literal


@dataclass
class DefenseConfig:
    """서버의 방어 설정"""

    enabled: bool = False
    """방어 활성화 여부"""

    type: Literal["norm_clipping", "coordinate_median"] | None = None
    """방어 방식 (norm_clipping, coordinate_median)"""

    max_norm: float = 1.0
    """노름 클리핑 상한"""

    aggregation_method: Literal["fedavg", "coordinate_median"] = "fedavg"
    """글로벌 모델 업데이트 집계 방식 (fedavg, coordinate_median)"""


@dataclass
class ServerConfig:
    """연합학습 서버 설정"""

    host: str
    """서버 호스트 주소 (예: 127.0.0.1)"""

    port: int
    """서버 포트 번호 (예: 5000)"""

    num_clients: int
    """등록된 클라이언트 수"""

    num_rounds: int
    """실행할 전체 학습 라운드 수"""

    participation_rate: float
    """라운드별 클라이언트 참여 비율 (0.0 ~ 1.0)"""

    participation_timeout: float = 10.0
    """클라이언트 참여 응답 대기 시간 (초)"""

    round_timeout: float = 60.0
    """라운드 학습 및 업데이트 제출 마감 시간 (초)"""

    defense: DefenseConfig = field(default_factory=DefenseConfig)
    """서버 방어 설정"""
