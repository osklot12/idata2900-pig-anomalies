from abc import ABC, abstractmethod
from typing import List

import numpy as np

from src.models.prediction import Prediction


class Predictor(ABC):
    """Predicts bounding boxes for images."""

    @abstractmethod
    def predict(self, image: np.ndarray) -> List[Prediction]:
        """
        Makes predictions for an image.

        Args:
            image (np.ndarray): the image to make a prediction for

        Returns:
            List[Prediction]: list of predictions
        """
        raise NotImplementedError