from typing import List, Tuple
import numpy as np
import torch

from src.data.dataclasses.annotated_frame import AnnotatedFrame
from src.models.converters.static_bbox_scaler import StaticBBoxScaler


class YOLOv8BatchConverter:
    """Converts batches of AnnotatedFrame into the format expected by Ultralytics YOLOv8."""

    @staticmethod
    def convert(batch: List[AnnotatedFrame]) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        images = []
        targets = []

        for idx, frame in enumerate(batch):
            img = frame.frame
            height, width = img.shape[:2]

            img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0  # (3, H, W)
            images.append(img_tensor)

            bbox_scaler = StaticBBoxScaler(width, height)

            target = []
            for ann in frame.annotations:
                scaled_bbox = bbox_scaler.scale(ann.bbox)
                cx = (scaled_bbox.x + scaled_bbox.width / 2) / width
                cy = (scaled_bbox.y + scaled_bbox.height / 2) / height
                w = scaled_bbox.width / width
                h = scaled_bbox.height / height
                cls = ann.cls.value
                target.append([idx, cls, cx, cy, w, h])

            targets.append(torch.tensor(target, dtype=torch.float32) if target else torch.zeros((0, 6), dtype=torch.float32))

        return torch.stack(images, dim=0), targets
