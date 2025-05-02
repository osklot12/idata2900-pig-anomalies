from typing import List
import numpy as np
import torch
from ultralytics.utils.ops import non_max_suppression
from ultralytics import YOLO

from src.models.prediction import Prediction
from src.models.predictor import Predictor


class YOLOXIPredictor(Predictor):
    """
    A Predictor interface for Ultralytics YOLOv11 (XI), using manual forward + NMS.
    """

    def __init__(self, model: YOLO, device: str = "cuda", conf_thres: float = 0.4, iou_thres: float = 0.65):
        self.model = model
        self.device = device
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres

    def predict(self, image: np.ndarray) -> List[Prediction]:
        tensor_img = (
            torch.from_numpy(image)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .float()
            .to(self.device) / 255.0
        )

        with torch.no_grad():
            raw_outputs = self.model(tensor_img)[0]  # Get raw predictions (before NMS)
            nms_preds = non_max_suppression(
                raw_outputs,
                conf_thres=self.conf_thres,
                iou_thres=self.iou_thres
            )[0]

        if nms_preds is None or len(nms_preds) == 0:
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
            for det in nms_preds.cpu()
        ]