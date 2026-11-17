from dataclasses import dataclass
from common.types import ModelWeights


@dataclass
class ClientUpdate:
    # 업데이트를 전송한 클라이언트 ID
    client_id: int

    # 업데이트가 속한 라운드
    round_id: int

    # 업데이트 계산에 사용한 글로벌 모델 버전
    model_version: int

    # 학습에 사용한 데이터 개수
    num_samples: int

    # 클라이언트가 계산한 모델 파라미터 업데이트 값
    update: ModelWeights

    # 업데이트의 L2 Norm
    update_norm: float = 0.0
