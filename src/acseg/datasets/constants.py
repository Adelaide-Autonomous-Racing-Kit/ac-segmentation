from dataclasses import dataclass
from typing import List, Tuple, Union


@dataclass
class ClassInformation:
    name: str
    train_id: int
    colour: Tuple[int, int, int]


MONZA_SEG_CLASSES = [
    ClassInformation("void", 0, (0, 0, 0)),
    ClassInformation("drivable", 1, (0, 255, 249)),
    ClassInformation("road", 2, (84, 84, 84)),
    ClassInformation("curb", 3, (255, 119, 51)),
    ClassInformation("track_limit", 4, (255, 255, 255)),
    ClassInformation("sand", 5, (255, 255, 0)),
    ClassInformation("grass", 6, (170, 255, 128)),
    ClassInformation("vehicle", 7, (255, 42, 0)),
    ClassInformation("structure", 8, (153, 153, 255)),
    ClassInformation("people", 9, (255, 179, 204)),
]


@dataclass
class DatasetInformation:
    n_classes: int
    size: Tuple[int, int]
    ignore_index: int
    class_labels: List[ClassInformation]
    noramlisation: Union[Tuple[Tuple[float], Tuple[float]], None]


# Stores information for implemented dataset, normalisation
# statistics calculated across training and validation sets
DATASET_INFO = {
    "monza": DatasetInformation(
        10,
        (736, 1280),
        -100,
        MONZA_SEG_CLASSES,
        None,
    ),
}
