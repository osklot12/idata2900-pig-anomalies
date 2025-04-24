# src/models/twod/yolo/viii/streaming_evaluator_viii.py

import numpy as np
import torch
from tqdm import tqdm
from typing import Dict, List, Any

from ultralytics.utils.ops import non_max_suppression
from tests.utils.yolox_batch_visualizer import YOLOXBatchVisualizer  # Use your visualizer

POSTPROCESS_CONF_THRESH = 0.001
POSTPROCESS_IOU_THRESH = 0.65


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
        global_image_idx = 0

        for batch in tqdm(self._dataloader, desc="Evaluating"):
            imgs = batch["img"].to(self._device).float()
            if imgs.ndim == 3:
                imgs = imgs.unsqueeze(0)

            cls = batch["cls"]
            bboxes = batch["bboxes"]
            batch_idx = batch["batch_idx"]

            print(f"✅ BATCH IMG SHAPE: {imgs.shape}")

            targets = []
            for i in range(len(imgs)):
                boxes = bboxes[batch_idx == i]
                labels = cls[batch_idx == i].float()
                if len(boxes):
                    boxes = boxes.clone()
                    boxes[:, :2] -= boxes[:, 2:] / 2  # cxcywh → xyxy
                    boxes[:, 2:] += boxes[:, :2]
                    merged = torch.cat([labels.unsqueeze(1), boxes], dim=1)
                    targets.append(merged.cpu().numpy())
                else:
                    targets.append(np.zeros((0, 5)))

            with torch.no_grad():
                preds = self._model(imgs)  # NO `augment=False` here!
                preds = preds[0] if isinstance(preds, (tuple, list)) else preds
                preds = non_max_suppression(preds, conf_thres=POSTPROCESS_CONF_THRESH, iou_thres=POSTPROCESS_IOU_THRESH)

            detections = [p.cpu().numpy() if p is not None else np.zeros((0, 6), dtype=np.float32) for p in preds]

            # ✅ Visualize predictions
            has_predictions = [len(p) > 0 for p in detections]
            if any(has_predictions):
                YOLOXBatchVisualizer.visualize_with_predictions(
                    images=imgs[has_predictions].cpu(),
                    targets=torch.tensor(targets)[has_predictions],
                    predictions=[p for i, p in enumerate(detections) if has_predictions[i]],
                    class_names=["tail_biting", "ear_biting", "belly_nosing", "tail down"],
                    start_idx=global_image_idx,
                    save_dir="./eval_visuals"
                )
                global_image_idx += sum(has_predictions)

            all_detections.extend(detections)
            all_annotations.extend(targets)

        from src.utils.eval_metrics import compute_stats_from_dets
        return compute_stats_from_dets(all_detections, all_annotations, self._num_classes, self._iou_thresh)
