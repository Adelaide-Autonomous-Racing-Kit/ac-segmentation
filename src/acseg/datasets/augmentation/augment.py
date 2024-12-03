import argparse
from typing import Tuple

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from loguru import logger

from acseg.datasets.segmentation import get_dataset_information


class Augmentor:
    def __init__(self, args: argparse.ArgumentParser):
        ds_info = get_dataset_information(args.dataset_name)
        transforms = [
            A.Resize(
                ds_info.size[0],
                ds_info.size[1],
                interpolation=cv2.INTER_CUBIC,
                mask_interpolation=cv2.INTER_NEAREST,
            ),
            A.ColorJitter(0.2, 0.2, 0.2, 0.2),
            A.ToGray(p=0.2),
            A.HorizontalFlip(p=0.5),
            A.Rotate(15.0, mask_interpolation=cv2.INTER_NEAREST, p=0.5),
        ]
        if ds_info.noramlisation is not None:
            transforms.append(A.Normalize(*ds_info.noramlisation))
        transforms.append(ToTensorV2())
        self._transforms = A.Compose(transforms, is_check_shapes=False)

    def __call__(self, image: np.array, target: np.array) -> Tuple[torch.Tensor]:
        augmented = self._transforms(image=image, mask=target)
        image = augmented["image"].to(torch.float32)
        return image, augmented["mask"]
