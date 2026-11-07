from typing import TypeAlias

import torch

# PyTorch 모델의 파라미터 이름과 Tensor를 매핑하는 타입
ModelWeights: TypeAlias = dict[str, torch.Tensor]

# PyTorch 모델의 Gradient 이름과 Tensor를 매핑하는 타입
ModelGradients: TypeAlias = dict[str, torch.Tensor]
