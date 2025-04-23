# src/models/twod/yolo/viii/streaming_evaluator_viii.py

import numpy as np
import torch
from tqdm import tqdm
from typing import Dict, List, Any

from ultralytics.utils.ops import non_max_suppression
from tests.utils.yolox_batch_visualizer import YOLOXBatchVisualizer


class StreamingEvaluatorVIII:
    def __init__(self, model, dataloader, device, num_classes, iou_thresh=0.5):
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
            imgs = batch["img"].float().to(self._device, non_blocking=True)
            if imgs.ndim == 3:
                imgs = imgs.unsqueeze(0)
            cls = batch["cls"]
            bboxes = batch["bboxes"]
            batch_idx = batch["batch_idx"]

            print("✅ BATCH IMG SHAPE:", imgs.shape)

            targets = []
            for i in range(len(imgs)):
                boxes = bboxes[batch_idx == i]
                labels = cls[batch_idx == i].float()
                if len(boxes):
                    boxes = boxes.clone()
                    boxes[:, :2] -= boxes[:, 2:] / 2  # xy_center → top-left
                    boxes[:, 2:] += boxes[:, :2]      # width/height → bottom-right
                    merged = torch.cat([labels.unsqueeze(1), boxes], dim=1)  # (cls, x1, y1, x2, y2)
                    targets.append(merged.cpu().numpy())
                else:
                    targets.append(np.zeros((0, 5)))

            with torch.no_grad():
                out = self._model(imgs)  # <-- NOT a dict
                preds = out[0] if isinstance(out, (tuple, list)) else out
                preds = non_max_suppression(preds, conf_thres=0.001, iou_thres=0.65)

            detections = []
            for pred in preds:
                if pred is None:
                    detections.append(np.zeros((0, 6), dtype=np.float32))
                else:
                    detections.append(pred.cpu().numpy())

            all_detections.extend(detections)
            all_annotations.extend(targets)

            # Visual debug (optional)
            # YOLOXBatchVisualizer.visualize_with_predictions(...)

        from src.utils.eval_metrics import compute_stats_from_dets

        metrics = compute_stats_from_dets(all_detections, all_annotations, self._num_classes, self._iou_thresh)
        return metrics
