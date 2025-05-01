import numpy as np


class PrecisionCalculator:
    """Calculates precision given a confusion matrix."""

    @staticmethod
    def calculate(confuse: np.ndarray, cls: int) -> float:
        """
        Calculates precision for a given class.

        Args:
            confuse (np.ndarray): confusion matrix
            cls (int): class to calculate precision for
        """
        tp = confuse[cls][cls]
        fp = np.sum(confuse[:, cls]) - confuse[cls, cls]

        return tp / (tp + fp)