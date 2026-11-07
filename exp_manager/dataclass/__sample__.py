# 실험 설정 생성
from exp_manager.dataclass import (
    ExpConfig,
    AttackConfig,
    DefenseConfig,
    EnvironmentConfig,
    ClientInfo,
    ExpResult,
)

config = ExpConfig(
    experiment_id="exp_001",
    experiment_type="federated",
    seed=42,
    num_clients=10,
    num_rounds=10,
    participation_rate=0.5,
    attack=AttackConfig(
        enabled=True, type="insider_collusion", malicious_clients=[1, 3], scale=-5.0
    ),
    defense=DefenseConfig(enabled=True, type="coordinate_median"),
    environment=EnvironmentConfig(
        delay_clients=[2, 5], delay_seconds=3.0, data_distribution="non_iid"
    ),
)


# 클라이언트 정보 생성
client = ClientInfo(
    client_id=0,
    host="127.0.0.1",
    port=5001,
    url="http://127.0.0.1:5001",
    malicious=False,
)


# 실험 결과 생성
result = ExpResult(experiment_id=config.experiment_id)
