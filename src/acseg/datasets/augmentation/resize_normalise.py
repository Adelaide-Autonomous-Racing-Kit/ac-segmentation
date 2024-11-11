import argparse
from typing import List, Tuple

from PIL import Image
from acseg.datasets.segmentation import get_dataset_information
import torch
from torchvision import transforms
from torchvision.transforms import InterpolationMode


class ResizeNormalise:
    def __init__(self, args: argparse.ArgumentParser):
        ds_info = get_dataset_information(args.dataset_name)
        pytorch_transforms = [
            transforms.Resize(
                tuple(ds_info.size),
                InterpolationMode.NEAREST,
            ),
            transforms.ToTensor(),
        ]
        self._preprep = transforms.Compose(pytorch_transforms)
        if ds_info.noramlisation is None:
            self._normalise = None
        else:
            self._normalise = transforms.Normalize(*ds_info.noramlisation)

    def __call__(self, image: Image, target: Image) -> Tuple[torch.Tensor]:
        image = self._preprep(image)
        if self._normalise is not None:
            image = self._normalise(image)
        target = self._preprep(target)
        return image, target


class Augmentor:
    def __init__(self, args: argparse.ArgumentParser):
        self._resize_and_normalise = ResizeNormalise(args)
        input_transforms = [
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.2),
            transforms.RandomGrayscale(0.2),
        ]
        self._input_transforms = transforms.Compose(input_transforms)
        self._p_horizontal_flip = 0.5
        self._shared_transforms = [
            (
                transforms.RandomRotation(10.0),
                transforms.functional.rotate,
            )
        ]

    def __call__(self, image: Image, target: Image) -> Tuple[torch.Tensor]:
        image, target = self._resize_and_normalise(image, target)
        image = self._input_transforms(image)
        image, target = self._maybe_horizontal_flip(image, target)
        image, target = self._apply_shared_transforms(image, target)
        return image, target

    def _maybe_horizontal_flip(
        self,
        image: torch.Tensor,
        target: torch.Tensor,
    ) -> Tuple[torch.Tensor]:
        if torch.rand(1) < self._p_horizontal_flip:
            image = transforms.functional.hflip(image)
            target = transforms.functional.hflip(target)
        return image, target

    def _apply_shared_transforms(
        self,
        image: torch.Tensor,
        target: torch.Tensor,
    ) -> Tuple[torch.Tensor]:
        for transform in self._shared_transforms:
            params = transform[0].get_params(transform[0].degrees)
            image = transform[1](image, params)
            target = transform[1](target, params)
        return image, target
