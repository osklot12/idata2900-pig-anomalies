# src/models/twod/yolo/viii/streaming_evaluator_viii.py

import numpy as np
import torch
from tqdm import tqdm
from typing import Dict, List, Any

from ultralytics.utils.ops import non_max_suppression

from src.models.twod.yolo.viii.viii_postprocess import postprocess
from src.models.twod.yolo.viii.viii_pred_visualizer import YOLOv8BatchVisualizer
from tests.utils.yolox_batch_visualizer import YOLOXBatchVisualizer  # Use your visualizer

POSTPROCESS_CONF_THRESH = 0.4
POSTPROCESS_IOU_THRESH = 0.65


class StreamingEvaluatorVIII:
    def __init__(self, model, dataloader, device, num_classes, iou_thresh=0.5, writer=None, epoch=0):
        self._model = model
        self._dataloader = dataloader
        self._device = device
        self._num_classes = num_classes
        self._iou_thresh = iou_thresh
        self._writer = writer
        self._epoch = epoch

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
                    boxes[:, :2] -= boxes[:, 2:] / 2
                    boxes[:, 2:] += boxes[:, :2]
                    boxes[:, [0, 2]] *= imgs.shape[3]  # x1, x2
                    boxes[:, [1, 3]] *= imgs.shape[2]  # y1, y2
                    merged = torch.cat([labels.unsqueeze(1), boxes], dim=1)
                    targets.append(merged.cpu().numpy())
                else:
                    targets.append(np.zeros((0, 5)))

            # 🔧 YOLOv11 raw output postprocess
            with torch.no_grad():
                raw_preds = self._model(imgs)
                results = non_max_suppression(
                    prediction=raw_preds,
                    conf_thres=POSTPROCESS_CONF_THRESH,
                    iou_thres=POSTPROCESS_IOU_THRESH,
                    multi_label=False
                )

            preds = []
            for det in results:
                if det is not None and len(det):
                    preds.append(det.cpu())
                else:
                    preds.append(torch.zeros((0, 6)))

            detections = [p.numpy() if p is not None else np.zeros((0, 6), dtype=np.float32) for p in preds]

            # ✅ Visualize predictions
            has_predictions = [len(p) > 0 for p in detections]
            if any(has_predictions):
                mask = torch.tensor(has_predictions, dtype=torch.bool, device=imgs.device)
                YOLOv8BatchVisualizer.visualize_with_predictions(
                    images=imgs[mask].cpu(),
                    targets=[targets[i] for i, keep in enumerate(has_predictions) if keep],
                    predictions=[p for i, p in enumerate(detections) if has_predictions[i]],
                    class_names=["tail_biting", "ear_biting", "belly_nosing", "tail_down"],
                    start_idx=global_image_idx,
                    save_dir="./eval_visuals"
                )
                global_image_idx += mask.sum().item()

            all_detections.extend(detections)
            all_annotations.extend(targets)

            print("🧪 Prediction counts:", [len(p) if p is not None else 0 for p in preds])

        from src.utils.eval_metrics import compute_stats_from_dets
        metrics = compute_stats_from_dets(all_detections, all_annotations, self._num_classes, self._iou_thresh)
        print("📊 Confusion Matrix:\n", metrics["confusion_matrix"])

        if self._writer:
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    self._writer.add_scalar(f"val/{k}", v, self._epoch)

        return {k: v for k, v in metrics.items() if isinstance(v, (int, float))}
