"""Bridge for role 3, which owns HTTP/download/background execution."""

from dataclasses import replace
from pathlib import Path
from .artifacts import load_checkpoint
from .config import DataConfig, ModelConfig, RuntimeConfig, TrainingConfig
from .contracts import RoundContext, TrainRequest
from .training import LocalTrainingResult, train_local


def resolve_training_config(defaults: TrainingConfig, request: TrainRequest) -> TrainingConfig:
    """API epochs/batch/lr override defaults; optimizer options stay shared."""
    return replace(defaults, local_epochs=request.local_epochs, batch_size=request.batch_size,
                   learning_rate=request.learning_rate)


def train_from_checkpoint(checkpoint_path: str | Path, payload: dict, dataset, *,
                          client_id: int, experiment_id: str, data_config: DataConfig,
                          training_defaults: TrainingConfig, runtime: RuntimeConfig,
                          seed: int, on_epoch=None) -> LocalTrainingResult:
    """Call explicitly in a training job, after accepting /train.

    The Dataset must use data_config preprocessing. HTTP file downloads and
    JSON/multipart consistency checks belong to the role-3 adapter.
    """
    request = TrainRequest.from_api(payload, experiment_id=experiment_id)
    training_config = resolve_training_config(training_defaults, request)
    context = RoundContext(experiment_id, request.round_id, request.model_version, (client_id,))
    checkpoint = load_checkpoint(checkpoint_path)
    if checkpoint["experiment_id"] != experiment_id or checkpoint["model_version"] != request.model_version:
        raise ValueError("체크포인트의 실험/모델 버전이 학습 요청과 다릅니다.")
    preprocessing = checkpoint["preprocessing"]
    if list(preprocessing["normalization_mean"]) != list(data_config.normalization_mean) or list(preprocessing["normalization_std"]) != list(data_config.normalization_std):
        raise ValueError("학습 데이터 전처리가 전역 모델의 전처리와 다릅니다.")
    return train_local(checkpoint["model_state"], dataset, client_id=client_id, context=context,
                       model_config=ModelConfig(**checkpoint["model_config"]),
                       training_config=training_config, runtime=runtime, seed=seed, on_epoch=on_epoch)
