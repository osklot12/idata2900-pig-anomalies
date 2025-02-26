from enum import Enum, auto


class DatasetSplit(Enum):
    """An enumerations representing the different splits of the dataset."""

    TRAIN = auto()
    """Split of training data."""

    VAL = auto()
    """Split of validation data."""

    TEST = auto()
    """Split of test data."""