from dataclasses import dataclass
from common.types import ModelWeights
from common.model import create_model


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


def create_random_global_model(seed: int = 42, num_parameters: int = 32, model_config=None):
    """실제 CNN을 생성한다. num_parameters 인자는 기존 호출과의 호환을 위해 유지한다."""
    del num_parameters
    return create_model(seed, model_config)
