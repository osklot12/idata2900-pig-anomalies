import numpy as np
import torch
from tqdm import tqdm
from typing import Dict, List, Any

from src.models.twod.yolo.ultralytics.pred_visualizer import PredVisualizer

POSTPROCESS_CONF_THRESH = 0.4
POSTPROCESS_IOU_THRESH = 0.65


class StreamingEvaluatorXI:
    """
    Evaluation class for YOLOv11-style models using the Predictor interface.

    This class evaluates a model over a dataloader and computes detection metrics.
    It also supports optional TensorBoard logging and batch visualization.

    Args:
        model: Predictor-wrapped object detection model.
        dataloader: DataLoader yielding batches with 'img', 'cls', 'bboxes', and 'batch_idx'.
        device: Device on which evaluation is performed.
        num_classes: Number of object classes.
        iou_thresh (float): IoU threshold for computing mAP.
        writer: Optional TensorBoard writer for logging.
        epoch (int): Current epoch (used in logging).
    """

    def __init__(self, model, dataloader, device, num_classes, iou_thresh=0.5, writer=None, epoch=0):
        self._model = model
        self._dataloader = dataloader
        self._device = device
        self._num_classes = num_classes
        self._iou_thresh = iou_thresh
        self._writer = writer
        self._epoch = epoch

    def evaluate(self) -> Dict[str, Any]:
        """
        Runs evaluation over the dataset and computes metrics.

        Returns:
            Dictionary of evaluation results, including mAP and per-class scores.
        """
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

            print(f"BATCH IMG SHAPE: {imgs.shape}")

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

            preds = []
            for img_tensor in imgs:
                img_np = (img_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                predictions = self._model.predict(img_np)
                pred_array = np.array(
                    [[p.x1, p.y1, p.x2, p.y2, p.conf, p.cls] for p in predictions],
                    dtype=np.float32
                )
                preds.append(pred_array)

            detections = [p if p is not None else np.zeros((0, 6), dtype=np.float32) for p in preds]

            # Visualize predictions
            has_predictions = [len(p) > 0 for p in detections]
            if any(has_predictions):
                mask = torch.tensor(has_predictions, dtype=torch.bool, device=imgs.device)
                PredVisualizer.visualize_with_predictions(
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

            print("Prediction counts:", [len(p) if p is not None else 0 for p in preds])

        from src.utils.eval_metrics import compute_stats_from_dets
        metrics = compute_stats_from_dets(
            all_detections,
            all_annotations,
            self._num_classes,
            self._iou_thresh
        )

        if self._writer:
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    self._writer.add_scalar(f"val/{k}", v, self._epoch)

        return {k: v for k, v in metrics.items() if isinstance(v, (int, float))}
