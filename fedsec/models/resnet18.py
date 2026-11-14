"""CIFAR-10의 32×32 입력에 맞춘 ResNet-18."""

import torch
from torch import nn

RESNET18_FORMAT = "resnet18_cifar10_v1"


def _remove_batch_counter(module, state, prefix, *args):
    # 일부 BatchNorm 체크포인트를 읽을 때 PyTorch가 추가하는 카운터를 제거한다.
    state.pop(prefix + "num_batches_tracked", None)


def create_resnet18(seed: int, dropout: float = 0.0) -> nn.Module:
    from torchvision.models import resnet18

    torch.random.default_generator.manual_seed(seed)
    model = resnet18(weights=None, num_classes=10)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    nn.init.kaiming_normal_(model.conv1.weight, mode="fan_out", nonlinearity="relu")
    model.maxpool = nn.Identity()
    model.avgpool = nn.AvgPool2d(kernel_size=4)
    if dropout:
        model.avgpool = nn.Sequential(model.avgpool, nn.Dropout(dropout))
    for layer in model.modules():
        if isinstance(layer, nn.BatchNorm2d):
            # 이동평균과 분산을 유지하고, 고정 momentum에 필요 없는 정수 카운터는 제외한다.
            layer.register_buffer("num_batches_tracked", None, persistent=False)
            layer.register_load_state_dict_pre_hook(_remove_batch_counter)
    return model
