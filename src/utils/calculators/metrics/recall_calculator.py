import numpy as np


class RecallCalculator:
    """Calculates recall given a confusion matrix."""

    @staticmethod
    def calculate(confuse: np.ndarray, cls: int) -> float:
        """
        Calculates recall for a given class.

        Args:
            confuse (np.ndarray): the confusion matrix
            cls (int): the class to calculate recall for
        """
        tp = confuse[cls, cls]
        fn = np.sum(confuse[cls, :]) - tp

        return tp / (tp + fn) if tp + fn > 0.0 else 0.0