from typing import List, Dict
import numpy as np
import torch

from src.data.dataclasses.annotated_frame import AnnotatedFrame


class UltralyticsBatchConverter:
    """Converts a batch into Ultralytics OBB training format."""

    @staticmethod
    def convert(batch: List[AnnotatedFrame]) -> List[Dict]:
        result = []

        for annotated_frame in batch:
            img = annotated_frame.frame  # HWC uint8

            # Skip invalid or empty images
            if img is None or img.size == 0:
                continue

            h, w = img.shape[:2]
            boxes = []
            classes = []

            for ann in annotated_frame.annotations:
                x = ann.bbox.x
                y = ann.bbox.y
                bw = ann.bbox.width
                bh = ann.bbox.height

                # Skip malformed boxes
                if bw <= 0 or bh <= 0:
                    continue

                cx = x + bw / 2
                cy = y + bh / 2
                angle = 0.0  # fixed angle for now

                boxes.append([cx, cy, bw, bh, angle])
                classes.append(ann.cls.value)

            # Skip frame if no valid annotations
            if len(boxes) == 0 or len(classes) == 0:
                continue

            result.append({
                "img": torch.tensor(img, dtype=torch.uint8),
                "instances": {
                    "bboxes": torch.tensor(boxes, dtype=torch.float32),
                    "cls": torch.tensor(classes, dtype=torch.int64),
                }
            })

        return result
