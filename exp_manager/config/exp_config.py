from dataclasses import dataclass, field
from typing import Literal


@dataclass
class AttackConfig:
    # 공격 활성화 여부
    enabled: bool = False

    # 공격 유형
    type: Literal["insider_collusion"] | None = None

    # 악의적인 클라이언트 ID 목록
    malicious_clients: list[int] = field(default_factory=list)

    # 악의적인 업데이트에 적용할 배율
    scale: float = -5.0


@dataclass
class DefenseConfig:
    # 방어 활성화 여부
    enabled: bool = False

    # 방어 유형
    type: Literal["norm_clipping", "coordinate_median"] | None = None

    # 방어 알고리즘별 추가 설정
    parameters: dict = field(default_factory=dict)


@dataclass
class EnvironmentConfig:
    # 지연시킬 클라이언트 ID 목록
    delay_clients: list[int] = field(default_factory=list)

    # 지연 시간(초)
    delay_seconds: float = 0.0

    # 데이터 분포 방식
    data_distribution: Literal["iid", "non_iid"] = "iid"


@dataclass
class ExpConfig:
    # 실험 식별자
    experiment_id: str

    # 실험 유형
    experiment_type: Literal["federated", "dlg"]

    # 랜덤 시드
    seed: int = 42

    # 전체 클라이언트 수
    num_clients: int = 10

    # 연합학습 라운드 수
    num_rounds: int = 10

    # 라운드별 클라이언트 참여 비율
    participation_rate: float = 0.5

    # 모델 관련 설정
    model_name: str = "CNN"
    num_classes: int = 10

    # 공격, 방어, 환경 설정
    attack: AttackConfig = field(default_factory=AttackConfig)
    defense: DefenseConfig = field(default_factory=DefenseConfig)
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
