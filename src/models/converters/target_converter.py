import torch
from typing import List
from src.models.converters.converter_interfaces import TargetToTensorConverter

class TargetConverter(TargetToTensorConverter):
    def __init__(self, device: torch.device):
        self.device = device

    def convert_to_tensors(self, targets: List[dict], images: List[torch.Tensor]) -> torch.Tensor:
        yolo_targets = []
        for i, t in enumerate(targets):
            boxes = t["boxes"]
            labels = t["labels"]
            _, img_h, img_w = images[i].shape  # C, H, W

            for box, label in zip(boxes, labels):
                x1, y1, x2, y2 = box
                xc = ((x1 + x2) / 2) / img_w
                yc = ((y1 + y2) / 2) / img_h
                w = (x2 - x1) / img_w
                h = (y2 - y1) / img_h
                yolo_targets.append([i, label.item(), xc.item(), yc.item(), w.item(), h.item()])

        return torch.tensor(yolo_targets, dtype=torch.float32, device=self.device)