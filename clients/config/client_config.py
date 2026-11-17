from dataclasses import dataclass, field
from typing import Literal


# 클라이언트의 공격 설정
@dataclass
class AttackConfig:

    # 공격 활성화 여부
    enabled: bool = False

    # 공격 종류 (insider_collusion)
    type: Literal["insider_collusion"] | None = None

    # 협력 대상 악성 클라이언트 ID
    malicious_client_id: int | None = None

    # 협력 대상 악성 클라이언트 URL
    malicious_client_url: str | None = None

    # 업데이트 변조 배율
    scale: float = -5.0


# 클라이언트 (로컬 서버) 설정
@dataclass
class ClientConfig:

    # 클라이언트 식별자
    client_id: int

    # 클라이언트 서버 호스트 주소 (예: 127.0.0.1)
    host: str

    # 클라이언트 서버 포트 번호 (예: 5001)
    port: int

    # 중앙 서버 API 기본 주소 (예: http://127.0.0.1:5000)
    server_url: str

    # 클라이언트 공격 설정
    attack: AttackConfig = field(default_factory=AttackConfig)
