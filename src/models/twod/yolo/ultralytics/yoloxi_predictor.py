from typing import List
import numpy as np
import torch
from ultralytics import YOLO

from src.models.prediction import Prediction
from src.models.predictor import Predictor


class YOLOXIPredictor(Predictor):
    """
    A Predictor interface for Ultralytics YOLOv11 (XI), using high-level `predict()` API.
    Applies confidence and IoU threshold filtering automatically.
    """

    def __init__(self, model: YOLO, device: str = "cuda", conf_thres: float = 0.4, iou_thres: float = 0.65):
        """
        Args:
            model (YOLO): The Ultralytics YOLO object (not .model)
            device (str): Device for inference ('cuda' or 'cpu')
            conf_thres (float): Confidence threshold for filtering
            iou_thres (float): IoU threshold for NMS
        """
        self.model = model
        self.device = device
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres

    def predict(self, image: np.ndarray) -> List[Prediction]:
        """
        Runs inference on a single image and returns filtered predictions.

        Args:
            image (np.ndarray): Input image in HWC uint8 format (0–255)

        Returns:
            List[Prediction]: List of structured predictions
        """
        tensor_img = (
            torch.from_numpy(image)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .float()
            .to(self.device) / 255.0
        )

        with torch.no_grad():
            results = self.model.predict(
                source=tensor_img,
                device=self.device,
                conf=self.conf_thres,
                iou=self.iou_thres,
                verbose=False
            )
            boxes = results[0].boxes

        if boxes is None or boxes.data.numel() == 0:
            return []

        return [
            Prediction(
                x1=float(b[0]),
                y1=float(b[1]),
                x2=float(b[2]),
                y2=float(b[3]),
                conf=float(b[4]),
                cls=int(b[5])
            )
            for b in boxes.data.cpu().numpy()
        ]
