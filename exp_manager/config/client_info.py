from dataclasses import dataclass


@dataclass
class ClientInfo:
    """클라이언트 서버 정보"""

    client_id: int
    """클라이언트 고유 ID (0 ~ num_clients - 1)"""

    host: str
    """클라이언트 서버 호스트 주소 (예: 127.0.0.1)"""

    port: int
    """클라이언트 서버 포트 번호 (예: 5001)"""

    url: str
    """클라이언트 API 기본 주소 (예: http://127.0.0.1:5001)"""

    malicious: bool = False
    """악성 클라이언트 여부"""
