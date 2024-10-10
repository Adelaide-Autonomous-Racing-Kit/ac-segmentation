import argparse

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
            self._normalise = transforms.Normalize(*ds_info.normalisation)

    def __call__(self, image: Image, target: Image) -> torch.Tensor:
        image = self._preprep(image)
        if self._normalise is not None:
            image = self._normalise(image)
        target = self._preprep(target)
        return image, target
