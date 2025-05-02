from mean_average_precision import MetricBuilder
import numpy as np

class MAPCalculator:
    def __init__(self, num_classes: int, iou_threshold: float = 0.5):
        self.num_classes = num_classes
        self.iou_threshold = iou_threshold
        self.metric = MetricBuilder.build_evaluation_metric(
            "map_2d", async_mode=False, num_classes=self.num_classes
        )

    def update(self, preds: np.ndarray, gts: np.ndarray):
        """
        Updates the metric with a batch of predictions and ground truths.

        Args:
            preds (np.ndarray): shape (N, 6) → [x1, y1, x2, y2, class_id, score]
            gts (np.ndarray): shape (M, 5) → [x1, y1, x2, y2, class_id]
        """
        if len(preds) == 0:
            preds = np.zeros((0, 6))
        if len(gts) == 0:
            gts = np.zeros((0, 5))

        # Padding since "difficult" and "crowd" is not used in our case (Either true or false)
        if gts.shape[1] == 5:
            padding = np.zeros((gts.shape[0], 2), dtype=np.float32)
            gts = np.hstack((gts, padding))

        self.metric.add(preds, gts)

    def compute(self):
        """Returns the final metric scores."""
        raw_result = self.metric.value(iou_thresholds=self.iou_threshold)
        iou_key = self.iou_threshold

        # Extract per-class APs
        class_results = raw_result.get(iou_key, {})
        per_class_ap = {
            cls_id: cls_data.get("ap", 0.0)
            for cls_id, cls_data in class_results.items()
        }

        return {
            "mAP": raw_result.get("mAP", 0.0),
            "per_class_ap": per_class_ap
        }

    def reset(self):
        """Clears the stored predictions and GTs."""
        self.metric.reset()
