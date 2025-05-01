from abc import ABC, abstractmethod

import numpy as np

from src.models.prediction import Prediction


class Predictor(ABC):
    """Predicts bounding boxes for images."""

    @abstractmethod
    def predict(self, image: np.ndarray) -> Prediction:
        """
        Makes a prediction for an image.

        Args:
            image (np.ndarray): the image to make a prediction for

        Returns:
            Prediction: the prediction
        """
        raise NotImplementedError