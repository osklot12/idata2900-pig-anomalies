from typing import List
import numpy as np
import torch
from src.models.prediction import Prediction
from src.models.predictor import Predictor


class YOLOXIPredictor(Predictor):
    """
    A Predictor interface implementation for Ultralytics YOLOv11 (XI).

    This class wraps a raw YOLO model and exposes a .predict(image) method
    that returns a list of `Prediction` objects from a NumPy image input.

    Args:
        model: Ultralytics model instance (e.g. YOLO or DetectionModel).
        device: torch.device to use for inference.
    """
    def __init__(self, model, device: torch.device):
        self.model = model
        self.device = device

    def predict(self, image: np.ndarray) -> List[Prediction]:
        """
        Runs inference on a single image and returns formatted predictions.

        Args:
            image (np.ndarray): Input image (HWC, uint8).

        Returns:
            List[Prediction]: List of bounding box predictions.
        """
        tensor_img = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float().to(self.device) / 255.0

        with torch.no_grad():
            preds = self.model(tensor_img)[0]

        predictions = []
        if preds is not None and len(preds) > 0:
            for det in preds.cpu():
                x1, y1, x2, y2, conf, cls = det.tolist()
                predictions.append(Prediction(x1, y1, x2, y2, conf, int(cls)))
        return predictions
