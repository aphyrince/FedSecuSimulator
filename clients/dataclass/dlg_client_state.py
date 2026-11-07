from dataclasses import dataclass
from typing import Any


@dataclass
class DLGClientState:
    """클라이언트의 DLG 실험 상태"""

    dlg_id: str
    """DLG 실험 고유 ID"""

    model_version: int
    """그래디언트 계산에 사용한 글로벌 모델 버전"""

    image_id: str
    """원본 이미지 식별자"""

    image_path: str
    """클라이언트 내부 원본 이미지 파일 경로"""

    label: int
    """원본 이미지의 정답 레이블"""

    noise_sigma: float
    """그래디언트에 적용할 가우시안 노이즈 표준편차"""

    gradient: Any = None
    """계산된 모델 그래디언트
    실제 자료형은 사용하는 딥러닝 프레임워크에 따라 달라짐
    """
