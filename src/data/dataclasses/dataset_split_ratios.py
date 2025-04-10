from dataclasses import dataclass

@dataclass(frozen=True)
class DatasetSplitRatios:
    """
    Split ratios for a dataset.

    Attributes:
        train (float): the ratio of training data
        val (float): the ratio of validation data
        test (float): the ratio of test data
    """
    train: float
    val: float
    test: float