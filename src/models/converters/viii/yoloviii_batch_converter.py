# src/models/converters/viii/yoloviii_batch_converter.py

from typing import List, Tuple
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

        for frame in batch:
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

            bboxes.append(torch.tensor(frame_bboxes, dtype=torch.float32))
            classes.append(torch.tensor(frame_classes, dtype=torch.long))

        return {
            "img": torch.stack(images),
            "bboxes": bboxes,
            "cls": classes,
        }
