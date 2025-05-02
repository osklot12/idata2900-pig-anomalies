from typing import List
import numpy as np
import torch
from ultralytics import YOLO
from src.models.prediction import Prediction
from src.models.predictor import Predictor


class YOLOXIPredictor(Predictor):
    """
    Predictor wrapper for Ultralytics YOLOv11 (XI).
    Uses the built-in .predict() method and parses Results into Prediction objects.
    """

    def __init__(self, model: YOLO, device: str = "cuda"):
        self.model = model
        self.device = device

    def predict(self, image: np.ndarray) -> List[Prediction]:
        tensor_img = (
            torch.from_numpy(image)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .float()
            .to(self.device) / 255.0
        )

        with torch.no_grad():
            results = self.model.predict(tensor_img)[0]

        if not hasattr(results, "boxes") or results.boxes.data.numel() == 0:
            return []

        return [
            Prediction(
                x1=float(det[0]),
                y1=float(det[1]),
                x2=float(det[2]),
                y2=float(det[3]),
                conf=float(det[4]),
                cls=int(det[5])
            )
            for det in results.boxes.data.cpu()
        ]
