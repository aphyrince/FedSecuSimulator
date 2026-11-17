from datetime import datetime

from clients.config import (
    ClientConfig,
    AttackConfig,
    ClientDataset,
    TrainingState,
    ClientUpdate,
    DLGClientState,
)

# ==========================================
# 1. 클라이언트 설정
# ==========================================

config = ClientConfig(
    client_id=0,
    host="127.0.0.1",
    port=5001,
    server_url="http://127.0.0.1:5000",
    attack=AttackConfig(enabled=False, type=None, malicious_clients=[], scale=-5.0),
)


# ==========================================
# 2. 클라이언트 데이터 정보
# ==========================================

dataset = ClientDataset(
    client_id=0,
    num_samples=5000,
    data_path="data/client_0/",
    class_distribution={
        0: 500,
        1: 500,
        2: 500,
        3: 500,
        4: 500,
        5: 500,
        6: 500,
        7: 500,
        8: 500,
        9: 500,
    },
)


# ==========================================
# 3. 학습 상태 생성
# ==========================================

training = TrainingState(
    round_id=1,
    model_version=0,
    status="training",
    num_samples=5000,
    local_epochs=5,
    batch_size=32,
    learning_rate=0.001,
    started_at=datetime.now(),
)


# ==========================================
# 4. 모델 업데이트 생성
# ==========================================

# 실제 구현에서는 학습이 끝난 후 계산한 업데이트를 넣습니다.
# 현재는 예시 데이터입니다.
dummy_update = [[0.01, 0.02, 0.03], [0.04, 0.05, 0.06]]

client_update = ClientUpdate(
    client_id=config.client_id,
    round_id=training.round_id,
    model_version=training.model_version,
    num_samples=training.num_samples,
    update=dummy_update,
    update_norm=0.095,
)


# ==========================================
# 5. 학습 종료
# ==========================================

training.status = "completed"
training.finished_at = datetime.now()


# ==========================================
# 6. DLG 클라이언트 상태 생성
# ==========================================

dlg_state = DLGClientState(
    dlg_id="dlg_001",
    model_version=0,
    image_id="cifar10_image_001",
    image_path="data/client_0/images/image_001.png",
    label=3,
    noise_sigma=0.01,
    gradient=[[0.01, 0.02, 0.03], [0.04, 0.05, 0.06]],
)


# ==========================================
# 7. 데이터 확인
# ==========================================

print("Client ID:", config.client_id)
print("Server URL:", config.server_url)

print("Dataset Samples:", dataset.num_samples)
print("Class Distribution:", dataset.class_distribution)

print("Training Status:", training.status)
print("Training Round:", training.round_id)

print("Update Norm:", client_update.update_norm)

print("DLG ID:", dlg_state.dlg_id)
print("DLG Image:", dlg_state.image_id)
