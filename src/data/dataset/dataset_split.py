from enum import Enum, auto


class DatasetSplit(Enum):
    """An enumeration representing the different splits of the dataset."""

    TRAIN = 0
    """Split of training data."""

    VAL = 1
    """Split of validation data."""

    TEST = 2
    """Split of test data."""