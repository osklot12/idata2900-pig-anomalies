import numpy as np
import torch
from tqdm import tqdm
from typing import Dict, List, Any

from ultralytics.utils.ops import non_max_suppression


class StreamingEvaluatorVIII:
    def __init__(self, model, dataloader, device, num_classes, iou_thresh=0.5):
        self._model = model
        self._dataloader = dataloader
        self._device = device
        self._num_classes = num_classes
        self._iou_thresh = iou_thresh

        self._wrap_concat_debug()

    def _wrap_concat_debug(self):
        """Wrap only Concat layers to log inputs. Avoids breaking Sequential or other core layers."""
        def wrap_forward(module):
            orig_forward = module.forward

            def new_forward(*args, **kwargs):
                if isinstance(args[0], (list, tuple)):
                    print(f"🧩 Concat at {module}: input = {[t.shape for t in args[0]]}")
                else:
                    print(f"🧩 Concat at {module}: input = {type(args[0])}")
                return orig_forward(*args, **kwargs)

            return new_forward

        for m in self._model.modules():
            if m.__class__.__name__ == "Concat":
                try:
                    m.forward = wrap_forward(m)
                except Exception as e:
                    print(f"⚠️ Could not wrap Concat layer: {m} — {e}")

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
            print("✅ BATCH IMG TYPE:", type(imgs))

            targets = []
            for i in range(len(imgs)):
                boxes = bboxes[batch_idx == i]
                labels = cls[batch_idx == i].float()
                if len(boxes) and len(labels):
                    boxes = boxes.clone()
                    boxes[:, :2] -= boxes[:, 2:] / 2  # xy_center → top-left
                    boxes[:, 2:] += boxes[:, :2]      # width/height → bottom-right
                    merged = torch.cat([labels.unsqueeze(1), boxes], dim=1)
                    targets.append(merged.cpu().numpy())
                else:
                    targets.append(np.zeros((0, 5), dtype=np.float32))

            with torch.no_grad():
                try:
                    out = self._model(imgs, augment=False)
                except TypeError:
                    out = self._model(imgs)

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

        from src.utils.eval_metrics import compute_stats_from_dets
        metrics = compute_stats_from_dets(all_detections, all_annotations, self._num_classes, self._iou_thresh)
        return metrics