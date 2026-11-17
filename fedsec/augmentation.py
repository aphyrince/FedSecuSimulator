"""CIFAR-10 로컬 학습에만 적용하는 이미지 증강."""

import torch
from torch.utils.data import Dataset

from fedsec.model import PREPROCESSING


AUGMENTATIONS = ("none", "flip", "crop", "flip_crop", "flip_crop_jitter", "flip_crop_cutout")


class AugmentedDataset(Dataset):
    def __init__(self, dataset: Dataset, augmentation: str):
        from torchvision import transforms

        self.dataset = dataset
        operations = []
        if augmentation.startswith("flip"):
            operations.append(transforms.RandomHorizontalFlip(p=0.5))
        if "crop" in augmentation:
            operations.append(transforms.RandomCrop(32, padding=4))
        if augmentation == "flip_crop_jitter":
            operations.append(transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2))
        self.transform = transforms.Compose(operations)
        self.cutout = augmentation == "flip_crop_cutout"
        self.mean = torch.tensor(PREPROCESSING["mean"]).reshape(3, 1, 1)
        self.std = torch.tensor(PREPROCESSING["std"]).reshape(3, 1, 1)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image, label = self.dataset[index]
        # 데이터 로더가 적용한 정규화를 풀어 픽셀 범위에서 증강한다.
        pixels = self.transform((image * self.std + self.mean).clamp(0, 1))
        if self.cutout:
            top, left = torch.randint(0, 25, (2,)).tolist()
            pixels[:, top:top + 8, left:left + 8] = 0
        return (pixels - self.mean) / self.std, label
