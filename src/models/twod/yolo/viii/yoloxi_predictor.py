from typing import List
import numpy as np
import torch

from src.models.prediction import Prediction
from src.models.predictor import Predictor
from ultralytics.utils.ops import non_max_suppression

POSTPROCESS_CONF_THRESH = 0.4
POSTPROCESS_IOU_THRESH = 0.65


class YOLOXIPredictor(Predictor):
    def __init__(self, model):
        self.model = model

    def predict(self, image: np.ndarray) -> List[Prediction]:
        """
        Makes predictions for a single image and returns a list of Prediction objects.

        Args:
            image (np.ndarray): Input image in HWC format

        Returns:
            List[Prediction]: Post-processed detections
        """
        tensor_img = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float().to(self.model.device) / 255.0
        with torch.no_grad():
            raw_output = self.model(tensor_img)

        nms_results = non_max_suppression(
            prediction=raw_output,
            conf_thres=POSTPROCESS_CONF_THRESH,
            iou_thres=POSTPROCESS_IOU_THRESH,
            multi_label=False
        )

        predictions = []
        det = nms_results[0]
        if det is not None and len(det):
            for box in det.cpu():
                x1, y1, x2, y2, conf, cls = box.tolist()
                predictions.append(Prediction(x1=x1, y1=y1, x2=x2, y2=y2, conf=conf, cls=int(cls)))

        return predictions
