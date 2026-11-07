from dataclasses import dataclass, field
from typing import Literal


@dataclass
class AttackConfig:
    """클라이언트의 공격 설정"""

    enabled: bool = False
    """공격 활성화 여부"""

    type: Literal["insider_collusion"] | None = None
    """공격 종류 (insider_collusion)"""

    malicious_clients: list[int] = field(default_factory=list)
    """악성 클라이언트 ID 목록"""

    scale: float = -5.0
    """업데이트 변조 배율"""


@dataclass
class ClientConfig:
    """클라이언트 서버 설정"""

    client_id: int
    """클라이언트 고유 ID"""

    host: str
    """클라이언트 서버 호스트 주소 (예: 127.0.0.1)"""

    port: int
    """클라이언트 서버 포트 번호 (예: 5001)"""

    server_url: str
    """중앙 서버 API 기본 주소 (예: http://127.0.0.1:5000)"""

    attack: AttackConfig = field(default_factory=AttackConfig)
    """클라이언트 공격 설정"""
