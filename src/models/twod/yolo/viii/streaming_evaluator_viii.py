import numpy as np
import torch
from tqdm import tqdm
from typing import Dict, List, Any

from ultralytics.utils.ops import non_max_suppression


class StreamingEvaluatorVIII:
    def __init__(self, model: torch.nn.Module, dataloader, device: torch.device, num_classes: int, iou_thresh=0.5):
        self._model = model
        self._dataloader = dataloader
        self._device = device
        self._num_classes = num_classes
        self._iou_thresh = iou_thresh

    def evaluate(self) -> Dict[str, Any]:
        self._model.eval()
        all_detections: List[np.ndarray] = []
        all_annotations: List[np.ndarray] = []

        for batch in tqdm(self._dataloader, desc="Evaluating"):
            imgs = batch["img"].float().to(self._device)
            if imgs.ndim == 3:
                imgs = imgs.unsqueeze(0)

            bboxes = batch["bboxes"]
            classes = batch["cls"]
            batch_idx = batch["batch_idx"]

            # === Prepare targets ===
            targets = []
            for i in range(len(imgs)):
                boxes = bboxes[batch_idx == i]
                labels = classes[batch_idx == i]
                if len(boxes) > 0:
                    boxes = boxes.clone()
                    boxes[:, :2] -= boxes[:, 2:] / 2  # center -> top-left
                    boxes[:, 2:] += boxes[:, :2]      # size -> bottom-right
                    merged = torch.cat([labels.unsqueeze(1), boxes], dim=1)
                    targets.append(merged.cpu().numpy())
                else:
                    targets.append(np.zeros((0, 5), dtype=np.float32))

            with torch.no_grad():
                preds = self._model(imgs)
                if isinstance(preds, (tuple, list)):
                    preds = preds[0]
                preds = non_max_suppression(preds, conf_thres=0.001, iou_thres=0.65)

            # === Process detections ===
            detections = []
            for pred in preds:
                if pred is None or pred.shape[0] == 0:
                    detections.append(np.zeros((0, 6), dtype=np.float32))
                else:
                    detections.append(pred.cpu().numpy())

            all_detections.extend(detections)
            all_annotations.extend(targets)

        from src.utils.eval_metrics import compute_stats_from_dets
        return compute_stats_from_dets(all_detections, all_annotations, self._num_classes, self._iou_thresh)
