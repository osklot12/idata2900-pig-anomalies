import torch
from typing import List, Dict
from torchmetrics.detection.mean_ap import MeanAveragePrecision


class TrainingMetricsCalculator:
    def __init__(self):
        self._map_metric = MeanAveragePrecision(iou_type="bbox")

    def calculate(self, predictions: List[Dict], targets: List[Dict]) -> Dict[str, float]:
        iou = self._calculate_iou(predictions, targets)
        map_ = self._calculate_map(predictions, targets)

        return {
            "IoU": iou,
            "mAP": map_["mAP"],
            "mAP@0.5": map_["mAP@0.5"],
            "Recall@100": map_["Recall@100"],
            "F1": map_["F1"]
        }

    def _calculate_iou(self, predictions: List[Dict], targets: List[Dict]) -> float:
        total_iou = 0.0
        match_count = 0

        for pred, target in zip(predictions, targets):
            pred_boxes = pred.get("boxes", torch.empty((0, 4)))
            target_boxes = target.get("boxes", torch.empty((0, 4)))
            matched = set()

            for pb in pred_boxes:
                best_iou, best_idx = 0.0, -1
                for idx, tb in enumerate(target_boxes):
                    if idx in matched:
                        continue
                    iou = self._box_iou(pb, tb)
                    if iou > best_iou:
                        best_iou, best_idx = iou, idx
                if best_idx != -1:
                    total_iou += best_iou
                    match_count += 1
                    matched.add(best_idx)

        return total_iou / match_count if match_count > 0 else 0.0

    def _calculate_map(self, predictions: List[Dict], targets: List[Dict]) -> Dict[str, float]:
        self._map_metric.reset()
        self._map_metric.update(preds=predictions, target=targets)
        result = self._map_metric.compute()

        precision = result.get("map_50", 0.0)
        recall = result.get("mar_100", 0.0)
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0 else 0.0
        )

        return {
            "mAP": float(result["map"]),
            "mAP@0.5": float(result["map_50"]),
            "Recall@100": float(result["mar_100"]),
            "F1": f1
        }

    def _box_iou(self, box1: torch.Tensor, box2: torch.Tensor) -> float:
        xA = torch.max(box1[0], box2[0])
        yA = torch.max(box1[1], box2[1])
        xB = torch.min(box1[2], box2[2])
        yB = torch.min(box1[3], box2[3])

        inter = torch.clamp(xB - xA, min=0) * torch.clamp(yB - yA, min=0)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - inter

        return float(inter / union) if union > 0 else 0.0
