from typing import List

import numpy as np


class ConfusionCalculator:
    """Calculates confusion matrices."""

    @staticmethod
    def calculate(predictions: List[int], targets: List[int], num_classes: int) -> np.ndarray:
        """
        Computes a confusion matrix given predictions and targets.

        Args:
            predictions (List[int]): the predictions
            targets (List[int]): the targets
            num_classes (int): the number of classes
        """
        A = np.zeros(shape=(num_classes + 1, num_classes + 1), dtype=int)
        for gt, pred in zip(targets, predictions):
            A[gt][pred] += 1

        return A