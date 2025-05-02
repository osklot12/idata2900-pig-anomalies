from typing import List
import numpy as np
import torch
from src.models.prediction import Prediction
from src.models.predictor import Predictor


class YOLOXIPredictor(Predictor):
    """
    Predictor wrapper for Ultralytics YOLOv11 model. Converts a NumPy image to a PyTorch tensor,
    runs inference, and returns predictions in a structured format.
    """

    def __init__(self, model):
        """
        Initializes the YOLOXIPredictor.

        Args:
            model: The Ultralytics YOLOv11 model (already loaded and moved to the appropriate device).
        """
        self.model = model

    def predict(self, image: np.ndarray) -> List[Prediction]:
        """
        Runs inference on a single image and returns structured predictions.

        Args:
            image (np.ndarray): Input image in HWC format with dtype uint8 or float32.

        Returns:
            List[Prediction]: A list of predicted bounding boxes, confidence scores, and class labels.
        """
        tensor_img = (
            torch.from_numpy(image)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .float()
            .to(self.model.device) / 255.0
        )

        with torch.no_grad():
            preds = self.model(tensor_img)[0]  # Ultralytics returns list of detections

        predictions = []
        if preds is not None and len(preds) > 0:
            for det in preds.cpu():
                x1, y1, x2, y2, conf, cls = det.tolist()
                predictions.append(Prediction(x1, y1, x2, y2, conf, int(cls)))

        return predictions
