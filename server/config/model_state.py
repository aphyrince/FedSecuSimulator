from dataclasses import dataclass


@dataclass
class ModelState:
    """서버의 글로벌 모델 상태"""

    model_version: int
    """현재 글로벌 모델 버전"""

    model_path: str
    """글로벌 모델 체크포인트 파일 경로"""

    architecture: str
    """모델 구조 이름 (예: ResNet18, CNN)"""

    num_parameters: int
    """모델의 전체 학습 가능한 파라미터 개수"""
