from dataclasses import dataclass, field


@dataclass
class ClientDataset:
    """클라이언트가 소유한 학습 데이터 정보"""

    client_id: int
    """데이터를 소유한 클라이언트 ID"""

    num_samples: int
    """학습 데이터 표본 수"""

    data_path: str
    """학습 데이터 저장 경로"""

    class_distribution: dict[int, int] = field(default_factory=dict)
    """클래스별 데이터 개수
    예: {
        0: 500,
        1: 300,
        2: 200
    }
    """
