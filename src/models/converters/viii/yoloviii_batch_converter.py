# src/models/converters/viii/yoloviii_batch_converter.py

from typing import List
import numpy as np
import torch
from src.data.dataclasses.annotated_frame import AnnotatedFrame
from src.models.converters.static_bbox_scaler import StaticBBoxScaler


class YOLOv8BatchConverter:
    """Converts batches of AnnotatedFrame into the format expected by Ultralytics YOLOv8."""

    @staticmethod
    def convert(batch: List[AnnotatedFrame]) -> dict:
        images = []
        bboxes = []
        classes = []
        batch_indices = []

        for idx, frame in enumerate(batch):
            img = frame.frame
            height, width = img.shape[:2]
            img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
            images.append(img_tensor)

            bbox_scaler = StaticBBoxScaler(width, height)

            frame_bboxes = []
            frame_classes = []

            for ann in frame.annotations:
                scaled_bbox = bbox_scaler.scale(ann.bbox)
                cx = (scaled_bbox.x + scaled_bbox.width / 2) / width
                cy = (scaled_bbox.y + scaled_bbox.height / 2) / height
                w = scaled_bbox.width / width
                h = scaled_bbox.height / height

                frame_bboxes.append([cx, cy, w, h])
                frame_classes.append(ann.cls.value)
                batch_indices.append(idx)  # ✨ one index per annotation

            bboxes.extend(frame_bboxes)
            classes.extend(frame_classes)

            for ann in frame.annotations:
                if ann.bbox.width < 1 or ann.bbox.height < 1:
                    print(f"⚠️ Suspicious bbox (very small or 0): {ann.bbox}")

        return {
            "img": torch.stack(images),
            "bboxes": torch.tensor(bboxes, dtype=torch.float32),
            "cls": torch.tensor(classes, dtype=torch.float32),
            "batch_idx": torch.tensor(batch_indices, dtype=torch.int64),
        }
