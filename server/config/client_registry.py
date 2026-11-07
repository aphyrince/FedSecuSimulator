from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal


@dataclass
class ClientRegistry:
    """서버에 등록된 클라이언트 정보"""

    client_id: int
    """클라이언트 고유 ID"""

    url: str
    """클라이언트 API 기본 주소 (예: http://127.0.0.1:5001)"""

    status: Literal["idle", "checking", "training", "offline"] = "offline"
    """클라이언트 현재 상태
    - idle: 대기 중
    - checking: 참여 가능 여부 확인 중
    - training: 학습 중
    - offline: 연결되지 않음
    """

    last_seen: datetime | None = None
    """클라이언트와 마지막으로 통신한 시각"""
