from dataclasses import dataclass, field
from typing import Any


@dataclass
class ClientUpdate:
    """클라이언트가 서버에 제출하는 로컬 모델 업데이트"""

    client_id: int
    """업데이트를 제출한 클라이언트 ID"""

    round_id: int
    """업데이트가 생성된 학습 라운드"""

    model_version: int
    """업데이트 생성에 사용한 글로벌 모델 버전"""

    num_samples: int
    """업데이트 생성에 사용한 학습 데이터 표본 수"""

    update: Any = None
    """글로벌 모델과 로컬 모델 사이의 가중치 차이
    실제 자료형은 사용하는 딥러닝 프레임워크에 따라 달라짐
    """

    update_norm: float = 0.0
    """업데이트 전체의 L2 노름"""
