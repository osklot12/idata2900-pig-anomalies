import numpy as np

class IoUCalculator:
    """Calculates IoU between predictions and ground truths."""

    @staticmethod
    def calculate(pred_boxes: np.ndarray, gt_boxes: np.ndarray) -> np.ndarray:
        """
        Compute IoU matrix between predicted boxes and ground truth boxes.

        Args:
            pred_boxes (np.ndarray): shape (N, 4) in format [x1, y1, x2, y2]
            gt_boxes (np.ndarray): shape (M, 4) in format [x1, y1, x2, y2]

        Returns:
            np.ndarray: IoU matrix of shape (N, M)
        """
        if len(pred_boxes) == 0 or len(gt_boxes) == 0:
            return np.zeros((len(pred_boxes), len(gt_boxes)))

        # Areas
        pred_areas = (pred_boxes[:, 2] - pred_boxes[:, 0]) * (pred_boxes[:, 3] - pred_boxes[:, 1])
        gt_areas = (gt_boxes[:, 2] - gt_boxes[:, 0]) * (gt_boxes[:, 3] - gt_boxes[:, 1])

        # Intersections
        inter_x1 = np.maximum(pred_boxes[:, None, 0], gt_boxes[:, 0])
        inter_y1 = np.maximum(pred_boxes[:, None, 1], gt_boxes[:, 1])
        inter_x2 = np.minimum(pred_boxes[:, None, 2], gt_boxes[:, 2])
        inter_y2 = np.minimum(pred_boxes[:, None, 3], gt_boxes[:, 3])
        inter_area = np.maximum(0, inter_x2 - inter_x1) * np.maximum(0, inter_y2 - inter_y1)

        # Unions
        union_area = pred_areas[:, None] + gt_areas - inter_area

        # IoU
        iou = inter_area / np.clip(union_area, a_min=1e-6, a_max=None)
        return iou