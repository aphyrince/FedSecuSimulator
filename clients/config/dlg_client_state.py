from dataclasses import dataclass
from common.types import ModelGradients


@dataclass
class DLGClientState:
    # DLG 공격 식별자
    dlg_id: str

    # 그래디언트 계산에 사용한 글로벌 모델 버전
    model_version: int

    # 원본 이미지 식별자
    image_id: str

    # 원본 이미지 경로
    image_path: str

    # 이미지 정답 레이블
    label: int

    # Gradient에 추가할 노이즈 표준편차
    noise_sigma: float

    # 계산한 Gradient
    gradient: ModelGradients | None = None
