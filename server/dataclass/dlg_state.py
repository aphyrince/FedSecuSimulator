from dataclasses import dataclass
from typing import Literal


@dataclass
class DLGState:
    """Deep Leakage from Gradients 실험 상태"""

    dlg_id: str
    """DLG 실험 고유 ID"""

    client_id: int
    """그래디언트를 제출할 클라이언트 ID"""

    model_version: int
    """DLG 공격에 사용할 글로벌 모델 버전"""

    noise_sigma: float
    """가우시안 노이즈 표준편차"""

    iterations: int
    """더미 이미지 최적화 반복 횟수"""

    status: Literal["pending", "running", "completed", "failed"] = "pending"
    """DLG 실험 진행 상태
    - pending: 실행 대기
    - running: 이미지 복원 진행 중
    - completed: 복원 완료
    - failed: 복원 실패
    """

    reconstructed_image_path: str | None = None
    """복원된 이미지 파일 저장 경로"""

    mse: float | None = None
    """원본 이미지와 복원 이미지 사이의 평균 제곱 오차"""
