import torch
from typing import List
from src.models.converters.converter_interfaces import TargetToTensorConverter

class TargetConverter(TargetToTensorConverter):
    def __init__(self, device: torch.device):
        self.device = device

    def convert_to_tensors(self, targets: List[dict], images: List[torch.Tensor]) -> List[torch.Tensor]:
        batch_targets = []

        for i, (target, image) in enumerate(zip(targets, images)):
            boxes = target["boxes"]
            labels = target["labels"]
            _, img_h, img_w = image.shape  # C, H, W

            yolo_target = []
            for box, label in zip(boxes, labels):
                x1, y1, x2, y2 = box
                xc = ((x1 + x2) / 2) / img_w
                yc = ((y1 + y2) / 2) / img_h
                w = (x2 - x1) / img_w
                h = (y2 - y1) / img_h
                yolo_target.append([label.item(), xc.item(), yc.item(), w.item(), h.item()])

            batch_targets.append(torch.tensor(yolo_target, dtype=torch.float32, device=self.device))

        return batch_targets
