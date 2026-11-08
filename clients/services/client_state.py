"""클라이언트 프로세스의 실험 상태를 관리한다."""

from typing import Any


class ClientState:
    """Flask route에서 사용하는 클라이언트 상태 모음."""

    def __init__(self) -> None:
        self.experiment_config: dict[str, Any] = {}
        self.dataset_range: dict[str, Any] = {}
        self.available = True
        self.global_model: dict[str, Any] = {}
        self.global_models: list[dict[str, Any]] = []
        self.local_training_results: list[dict[str, Any]] = []
        self.latest_local_training: dict[str, Any] = {}
        self.extracted_updates: dict[int, dict[str, Any]] = {}

    def set_experiment_config(self, config: dict[str, Any]) -> dict[str, Any]:
        self.experiment_config = config
        return self.experiment_config

    def set_dataset_range(self, dataset_range: dict[str, Any]) -> dict[str, Any]:
        self.dataset_range = dataset_range
        return self.dataset_range

    def receive_global_model(self, model: dict[str, Any]) -> dict[str, Any]:
        self.global_model = model
        self.global_models.append(model)
        return self.global_model

    def save_training(self, training_record: dict[str, Any]) -> None:
        round_id = int(training_record["round"])
        self.latest_local_training = training_record
        self.local_training_results.append(training_record)
        extracted_update = training_record.get("extracted_update")
        if isinstance(extracted_update, dict):
            self.extracted_updates[round_id] = extracted_update
