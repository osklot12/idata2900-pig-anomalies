from typing import List

import numpy as np
import torch
from torch.nn import Module

from src.models.prediction import Prediction
from src.models.predictor import Predictor
from torchvision.transforms.functional import normalize


class FasterRCNNPredictor(Predictor):
    """Predictor wrapper for faster-RCNN model."""

    def __init__(self, model: Module, device: torch.device, conf_thresh: float = 0.5):
        """
        Initializes a FasterRCNNPredictor instance.

        Args:
            model (Module): the model to use for prediction
            device (torch.device): the device to use for prediction
        """
        self._model = model
        self._device = device
        self._conf_thresh = conf_thresh

    def predict(self, image: np.ndarray) -> List[Prediction]:
        with torch.no_grad():
            img_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            img_tensor = normalize(
                img_tensor,
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ).to(self._device)
            outputs = self._model([img_tensor])[0]

            return [
                Prediction(
                    x1=float(b[0]), y1=float(b[1]), x2=float(b[2]), y2=float(b[3]), cls=int(c), conf=float(s)
                )
                for b, c, s in zip(outputs["boxes"], outputs["labels"], outputs["scores"])
                if float(s) >= self._conf_thresh
            ]