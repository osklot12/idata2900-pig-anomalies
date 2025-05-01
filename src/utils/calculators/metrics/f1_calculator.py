import numpy as np

from src.utils.calculators.metrics.precision_calculator import PrecisionCalculator
from src.utils.calculators.metrics.recall_calculator import RecallCalculator


class F1Calculator:
    """Calculates F1 score given a confusion matrix."""

    @staticmethod
    def calculate(confuse: np.ndarray, cls: int) -> float:
        """
        Calculates F1 score for the given class.

        Args:
            confuse (np.ndarray): the confusion matrix
            cls (int): the class to calculate F1 score for

        Returns:
            float: the f1 score
        """
        precision = PrecisionCalculator.calculate(confuse, cls)
        recall = RecallCalculator.calculate(confuse, cls)

        return 2 * precision * recall / (precision + recall) if precision + recall != 0 else 0.0