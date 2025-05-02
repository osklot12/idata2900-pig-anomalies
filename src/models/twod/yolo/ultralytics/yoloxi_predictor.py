from typing import List
import numpy as np
from ultralytics import YOLO

from src.models.prediction import Prediction
from src.models.predictor import Predictor


class YOLOXIPredictor(Predictor):
    """
    A Predictor interface for Ultralytics YOLOv11 (XI), using high-level `predict()` API.
    """

    def __init__(self, model: YOLO, device: str = "cuda", conf_thres: float = 0.4, iou_thres: float = 0.65):
        """
        Args:
            model (YOLO): The Ultralytics YOLO object (not .model)
            device (str): Device for inference ('cuda' or 'cpu')
            conf_thres (float): Confidence threshold
            iou_thres (float): IoU threshold for NMS
        """
        self.model = model
        self.device = device
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres

    def predict(self, image: np.ndarray) -> List[Prediction]:
        """
        Runs inference using Ultralytics' high-level API and returns structured predictions.

        Args:
            image (np.ndarray): Input image (HWC, uint8) in [0, 255]

        Returns:
            List[Prediction]: Postprocessed detections
        """
        results = self.model.predict(image, conf=self.conf_thres, iou=self.iou_thres, verbose=False)[0]

        if results is None or results.boxes is None or len(results.boxes) == 0:
            return []

        boxes = results.boxes

        return [
            Prediction(
                x1=float(xyxy[0]),
                y1=float(xyxy[1]),
                x2=float(xyxy[2]),
                y2=float(xyxy[3]),
                conf=float(conf),
                cls=int(cls)
            )
            for xyxy, conf, cls in zip(boxes.xyxy.cpu(), boxes.conf.cpu(), boxes.cls.cpu())
        ]
