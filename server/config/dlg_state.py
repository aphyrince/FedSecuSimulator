from dataclasses import dataclass
from typing import Literal
from common.types import ModelGradients


@dataclass
class DLGState:
    # DLG 공격 식별자
    dlg_id: str

    # 공격 대상 클라이언트 ID
    client_id: int

    # Gradient 계산에 사용한 모델 버전
    model_version: int

    # Gradient에 적용된 노이즈 표준편차
    noise_sigma: float

    # 최적화 반복 횟수
    iterations: int

    # 공격 진행 상태
    status: Literal["pending", "running", "completed", "failed"] = "pending"

    # 클라이언트에서 전달받은 Gradient
    gradient: ModelGradients | None = None

    # 복원된 이미지 저장 경로
    reconstructed_image_path: str | None = None
