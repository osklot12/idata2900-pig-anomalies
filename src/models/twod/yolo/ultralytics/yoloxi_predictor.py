from typing import List
import numpy as np
import torch
from ultralytics.utils.ops import non_max_suppression

from src.models.prediction import Prediction
from src.models.predictor import Predictor


class YOLOXIPredictor(Predictor):
    """
    A Predictor interface implementation for Ultralytics YOLOv11 (XI).

    Wraps a YOLOv11 model to return structured predictions via `predict(image)`.
    Applies postprocessing (NMS, conf threshold) to filter outputs.
    """

    def __init__(self, model, device: torch.device, conf_thres: float = 0.4, iou_thres: float = 0.65):
        """
        Args:
            model: Ultralytics YOLOv11 model (e.g. `YOLO(...)` or `.model`)
            device: The torch device to run inference on (e.g., cuda or cpu)
            conf_thres: Confidence threshold for filtering predictions
            iou_thres: IoU threshold for non-max suppression
        """
        self.model = model
        self.device = device
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres

    def predict(self, image: np.ndarray) -> List[Prediction]:
        """
        Runs inference on a single image and returns structured predictions.

        Args:
            image (np.ndarray): Input image (HWC, uint8) in range [0, 255]

        Returns:
            List[Prediction]: Filtered predictions with bounding boxes, confidence, and class
        """
        tensor_img = (
            torch.from_numpy(image)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .float()
            .to(self.device) / 255.0
        )

        with torch.no_grad():
            outputs = self.model(tensor_img)  # returns (preds,)
            preds = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
            nms_preds = non_max_suppression(preds, conf_thres=self.conf_thres, iou_thres=self.iou_thres)[0]

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
