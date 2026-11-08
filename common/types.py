from typing import TYPE_CHECKING, Any, TypeAlias

if TYPE_CHECKING:
    import torch

    Tensor = torch.Tensor
else:
    Tensor = Any

# PyTorch 모델의 파라미터 이름과 Tensor를 매핑하는 타입
ModelWeights: TypeAlias = dict[str, Tensor]

# PyTorch 모델의 Gradient 이름과 Tensor를 매핑하는 타입
ModelGradients: TypeAlias = dict[str, Tensor]
