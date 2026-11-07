from dataclasses import dataclass
from common.types import ModelWeights


@dataclass
class ModelState:
    # 현재 글로벌 모델 버전
    model_version: int

    # 글로벌 모델 체크포인트 파일 경로
    model_path: str

    # 모델 아키텍처 이름 (예: ResNet18, CNN)
    architecture: str

    # 전체 파라미터 개수
    num_parameters: int

    # 글로벌 모델 파라미터
    weights: ModelWeights | None = None
