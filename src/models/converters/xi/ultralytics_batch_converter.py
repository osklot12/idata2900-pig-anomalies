from typing import List, Dict
import numpy as np
import torch

from src.data.dataclasses.annotated_frame import AnnotatedFrame


class UltralyticsBatchConverter:
    """Converts a batch into Ultralytics OBB training format."""

    @staticmethod
    def convert(batch: List[AnnotatedFrame]) -> List[Dict[str, object]]:
        converted = []

        for i, annotated_frame in enumerate(batch):
            img = annotated_frame.frame  # shape [H, W, C], dtype=uint8
            h, w = img.shape[:2]

            # 👇 Convert to [C, H, W] and normalize to [0.0, 1.0] float32
            img_tensor = torch.tensor(img, dtype=torch.uint8).permute(2, 0, 1).float() / 255.0

            bboxes = []
            classes = []

            for ann in annotated_frame.annotations:
                x, y, bw, bh = ann.bbox.x, ann.bbox.y, ann.bbox.width, ann.bbox.height
                cx, cy = x + bw / 2, y + bh / 2
                angle = 0.0  # YOLO OBB expects 5-element box
                bboxes.append([cx, cy, bw, bh, angle])
                classes.append(ann.cls.value)

            sample = {
                "img": img_tensor,
                "instances": {
                    "bboxes": torch.tensor(bboxes, dtype=torch.float32) if bboxes else torch.empty((0, 5)),
                    "cls": torch.tensor(classes, dtype=torch.int64) if classes else torch.empty((0,), dtype=torch.int64),
                },
                "batch_idx": torch.full((len(bboxes),), i, dtype=torch.int64) if bboxes else torch.empty((0,), dtype=torch.int64),
            }

            converted.append(sample)

        return converted
