from typing import List

import numpy as np
import torch

from ext.yolox.yolox.utils import postprocess
from src.models.prediction import Prediction
from src.models.predictor import Predictor


class YOLOXPredictor(Predictor):
    """Predictor wrapper for YOLOX."""

    def __init__(self, model: torch.nn.Module, device: torch.device, conf_thresh: float = 0.5):
        self._model = model
        self._device = device
        self._conf_thresh = conf_thresh

    @torch.no_grad()
    def predict(self, image: np.ndarray) -> List[Prediction]:
        img_tensor = torch.from_numpy(image).float().permute(2, 0, 1).unsqueeze(0).to(self._device) / 255.0
        outputs = self._model(img_tensor)

        if isinstance(outputs, (list, tuple)):
            outputs = outputs[0]

        outputs = postprocess(outputs, num_classes=len(self._model.head.cls_preds), conf_thre=self._conf_thresh,
                              nms_thre=0.5)

        preds = []
        if outputs[0] is not None:
            for det in outputs[0].cpu().numpy():
                x1, y1, x2, y2, score, cls_id = det
                preds.append(
                    Prediction(
                        x1=float(x1),
                        y1=float(y1),
                        x2=float(x2),
                        y2=float(y2),
                        conf=float(score),
                        cls=int(cls_id)
                    )
                )

        return preds
