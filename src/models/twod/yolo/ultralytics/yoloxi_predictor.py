from typing import List
import numpy as np
import torch

from src.models.prediction import Prediction
from src.models.predictor import Predictor


class YOLOXIPredictor(Predictor):
    """
    Predictor wrapper for YOLOv11 models.

    This class handles preprocessing of input images, model inference,
    and conversion of outputs to a standardized `Prediction` format.

    Args:
        model: A YOLOv11 model instance with a `.device` attribute.
    """

    def __init__(self, model):
        self.model = model

    def predict(self, image: np.ndarray) -> List[Prediction]:
        """
        Performs inference on an image and returns standardized predictions.

        Args:
            image (np.ndarray): The input image in HWC format with dtype uint8 or float32.

        Returns:
            List[Prediction]: A list of predictions with bounding boxes, confidence scores, and class indices.
        """
        tensor_img = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float().to(self.model.device) / 255.0
        with torch.no_grad():
            preds = self.model(tensor_img)[0]  # YOLOv11 returns a list with a single tensor

        predictions = []
        if preds is not None and len(preds) > 0:
            for det in preds.cpu():
                x1, y1, x2, y2, conf, cls = det.tolist()
                predictions.append(Prediction(x1, y1, x2, y2, conf, int(cls)))

        return predictions
