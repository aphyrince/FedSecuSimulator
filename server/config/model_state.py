from dataclasses import dataclass
import random
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


def create_random_global_model(seed: int = 42, num_parameters: int = 32):
    """실험 시작용 기본 랜덤 전역 모델을 만든다."""
    rng = random.Random(seed)
    weights = {
        "layer_0.weight": [round(rng.uniform(-1.0, 1.0), 6) for _ in range(num_parameters)],
        "layer_0.bias": [round(rng.uniform(-1.0, 1.0), 6) for _ in range(max(1, num_parameters // 4))],
    }
    return {
        "model_version": 0,
        "architecture": "custom_random",
        "num_parameters": num_parameters,
        "weights": weights,
    }
