import argparse
from pathlib import Path
from typing import List

import numpy as np
import torch
import torchvision
from PIL import Image
from torch.utils.data import Dataset

from acseg.datasets.constants import DATASET_INFO, DatasetInformation


def get_dataset_information(dataset_name: str) -> DatasetInformation:
    return DATASET_INFO[dataset_name]


# Returns the dataset specified in the passed arguments
def build_dataset(
    args: argparse.Namespace,
    image_set: str = "train",
    transform: torchvision.transforms = None,
) -> Dataset:
    dataset = MonzaDataset(
        args.dataset_dir,
        train_transform=transform,
        val_transform=transform,
    )
    return dataset


# Abstract Class for implementing custom databases with
class CustomDataset(Dataset):
    # Returns the number of the current stages file pairs
    def __len__(self) -> int:
        return len(self._sample_filepaths[self._stage])

    # Identity function to be overridden in each dataset's implementation
    def _encode_target(self, target: torch.Tensor) -> torch.Tensor:
        return target

    # Returns the image and ground truth at the specified index
    # within the set files for the current stage
    def __getitem__(self, idx: int) -> torch.Tensor:
        image_path, target_path = self._sample_filepaths[self._stage][idx]
        image, target = Image.open(image_path), Image.open(target_path)
        image = self._maybe_convert_grey_image(image)
        image, target = np.array(image), np.array(target)
        image, target = self._apply_transforms(image, target)
        return image, target

    def _maybe_convert_grey_image(self, image: Image) -> Image:
        # Ensures black and white images can be batched with RGB
        if image.mode == "L":
            image = image.convert("RGB")
        return image

    def _apply_transforms(self, image: Image, target: Image) -> torch.Tensor:
        # Apply transforms to the image
        if self._transforms[self._stage]:
            if self._transforms[self._stage + "GT"]:
                image = self._transforms[self._stage](image)
                target = self._transforms[self._stage + "GT"](target)
            else:
                image, target = self._transforms[self._stage](image, target)
            # Encode loaded map to integer labels
            target = self._encode_target(target)
        return image, target

    # Mutator to swap stages
    def set_stage(self, stage: str):
        self._stage = stage


class MonzaDataset(CustomDataset):
    # Monza road and track limits dataset
    def __init__(
        self,
        root: str,
        train_transform: torchvision.transforms = None,
        train_transformGT: torchvision.transforms = None,
        val_transform: torchvision.transforms = None,
        val_transformGT: torchvision.transforms = None,
    ):
        # Store database root directory
        self._root = Path(root)
        # Store image transforms
        self._transforms = {
            "train": train_transform,
            "trainGT": train_transformGT,
            "val": val_transform,
            "valGT": train_transformGT,
        }
        # Flag for which filelist/transform to be used
        self._stage = "train"
        # Store list of image GT pairs for each section
        self._sample_filepaths = {
            "train": self._get_image_paths("train"),
            "val": self._get_image_paths("val"),
        }

    def _get_image_paths(self, stage: str) -> List[Path]:
        path = self._root.joinpath(stage)
        filenames = path.glob("*.jpeg")
        samples = {file.stem.split(".")[0] for file in filenames}
        file_pairs = [
            (
                path.joinpath(f"{sample}.jpeg"),
                path.joinpath(f"{sample}-ids.png"),
            )
            for sample in list(samples)
        ]
        return file_pairs

    def _encode_target(self, target: torch.Tensor) -> torch.LongTensor:
        target = target.long()
        target = target + 1
        target[target > 9] = 0
        return target
