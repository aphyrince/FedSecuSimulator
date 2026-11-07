from server.config import (
    ServerConfig,
    DefenseConfig,
    ClientRegistry,
    RoundState,
    ModelState,
    DLGState,
)

config = ServerConfig(
    host="127.0.0.1",
    port=5000,
    num_clients=10,
    num_rounds=100,
    participation_rate=0.5,
    participation_timeout=10.0,
    round_timeout=60.0,
    defense=DefenseConfig(
        enabled=True, type="norm_clipping", max_norm=1.0, aggregation_method="fedavg"
    ),
)


# 클라이언트 등록
clients: dict[int, ClientRegistry] = {}

clients[0] = ClientRegistry(client_id=0, url="http://127.0.0.1:5001", status="idle")


# 라운드 생성
round_state = RoundState(experiment_id="exp_001", round_id=1, model_version=0)


# 글로벌 모델 상태
model_state = ModelState(
    model_version=0,
    model_path="checkpoints/global_model.pth",
    architecture="ResNet18",
    num_parameters=11173962,
)


# DLG 실험 상태
dlg_state = DLGState(
    dlg_id="dlg_001", client_id=0, model_version=0, noise_sigma=0.01, iterations=300
)
