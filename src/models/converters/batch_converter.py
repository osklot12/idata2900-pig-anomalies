import torch
from typing import List, Dict
from src.models.converters.converter_interfaces import BatchToTupleConverter

class BatchConverter(BatchToTupleConverter):
    def __init__(self, device: torch.device):
        self.device = device

    def convert_to_tuple_of_tensors(self, batch) -> tuple[List[torch.Tensor], List[Dict]]:
        images = []
        targets = []
        for frame in batch:
            images.append(
                torch.tensor(frame.frame, dtype=torch.float32, device=self.device).permute(2, 0, 1) / 255.0
            )
            boxes = torch.tensor([
                [
                    ann.bbox.x,
                    ann.bbox.y,
                    ann.bbox.x + ann.bbox.width,
                    ann.bbox.y + ann.bbox.height
                ]
                for ann in frame.annotations
            ], dtype=torch.float32, device=self.device)
            labels = torch.tensor(
                [ann.cls.value for ann in frame.annotations],  # ✅ fix here
                dtype=torch.int64,
                device=self.device
            )
            targets.append({"boxes": boxes, "labels": labels})
        return images, targets
