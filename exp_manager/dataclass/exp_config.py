from dataclasses import dataclass, field
from typing import Literal


@dataclass
class AttackConfig:
    """공격 설정"""

    enabled: bool = False
    """공격 활성화 여부"""

    type: Literal["insider_collusion"] | None = None
    """공격 종류 (예: insider_collusion)"""

    malicious_clients: list[int] = field(default_factory=list)
    """악성 클라이언트 ID 목록"""

    scale: float = -5.0
    """업데이트 변조 배율 (기본값: -5.0)"""


@dataclass
class DefenseConfig:
    """방어 설정"""

    enabled: bool = False
    """방어 활성화 여부"""

    type: Literal["norm_clipping", "coordinate_median"] | None = None
    """방어 방식 (norm_clipping, coordinate_median)"""

    parameters: dict = field(default_factory=dict)
    """방어 방식별 세부 파라미터"""


@dataclass
class EnvironmentConfig:
    """실험 환경 설정"""

    delay_clients: list[int] = field(default_factory=list)
    """지연을 적용할 클라이언트 ID 목록"""

    delay_seconds: float = 0.0
    """클라이언트에 적용할 가상 지연 시간 (초)"""

    data_distribution: Literal["iid", "non_iid"] = "iid"
    """데이터 분포 방식 (iid, non_iid)"""


@dataclass
class ExpConfig:
    """실험 전체 설정"""

    experiment_id: str
    """실험 고유 ID"""

    experiment_type: Literal["federated", "dlg"]
    """실험 종류 (federated, dlg)"""

    seed: int
    """실험 난수 시드"""

    num_clients: int
    """전체 클라이언트 수"""

    num_rounds: int
    """전체 학습 라운드 수"""

    participation_rate: float
    """라운드별 클라이언트 참여 비율 (0.0 ~ 1.0)"""

    attack: AttackConfig = field(default_factory=AttackConfig)
    """공격 설정"""

    defense: DefenseConfig = field(default_factory=DefenseConfig)
    """방어 설정"""

    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    """실험 환경 설정"""
