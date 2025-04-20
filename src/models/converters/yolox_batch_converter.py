from typing import List, Tuple

import numpy as np
import torch

from src.data.dataclasses.annotated_frame import AnnotatedFrame
from src.models.converters.static_bbox_scaler import StaticBBoxScaler


class YOLOXBatchConverter:
    """Converts batches into the expected format for YOLOX.s"""

    @staticmethod
    def convert(batch: List[AnnotatedFrame]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        images = []
        targets = []
        img_info = []
        img_ids = []

        for idx, annotated_frame in enumerate(batch):
            img = annotated_frame.frame
            height, width = img.shape[:2]

            img = img.astype(np.float32) / 255.0
            img = np.transpose(img, (2, 0, 1))
            images.append(torch.tensor(img, dtype=torch.float32))

            bbox_scaler = StaticBBoxScaler(width, height)

            frame_targets = []
            for ann in annotated_frame.annotations:
                scaled_bbox = bbox_scaler.scale(ann.bbox)

                cx = scaled_bbox.x + scaled_bbox.width / 2
                cy = scaled_bbox.y + scaled_bbox.height / 2
                w = scaled_bbox.width
                h = scaled_bbox.height
                cls = ann.cls.value

                frame_targets.append([cls, cx, cy, w, h])

            if frame_targets:
                targets.append(torch.tensor(frame_targets, dtype=torch.float32))
            else:
                targets.append(torch.zeros((0, 5), dtype=torch.float32))

            img_info.append(torch.tensor([height, width, 1.0], dtype=torch.float32))
            img_ids.append(torch.tensor(idx, dtype=torch.int64))

        return (
            torch.stack(images, dim=0),
            YOLOXBatchConverter.pad_targets(targets),
            torch.stack(img_info, dim=0),
            torch.stack(img_ids, dim=0)
        )

    @staticmethod
    def pad_targets(targets: List[torch.Tensor]) -> torch.Tensor:
        """
        Pads variable-length targets into a single tensor of shape (B, max_boxes, 5)
        """
        batch_size = len(targets)
        max_boxes = max(t.shape[0] for t in targets)

        padded = torch.zeros((batch_size, max_boxes, 5), dtype=torch.float32)
        for i, t in enumerate(targets):
            if t.shape[0] > 0:
                padded[i, :t.shape[0], :] = t

        return padded
