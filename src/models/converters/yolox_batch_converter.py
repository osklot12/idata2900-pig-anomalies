from typing import List, Dict, Tuple

import numpy as np
import torch

from src.data.dataclasses.annotated_frame import AnnotatedFrame
from src.models.converters.bbox_to_corners import BBoxToCorners
from src.models.converters.static_bbox_scaler import StaticBBoxScaler


class YOLOXBatchConverter:
    """Converts batches into the expected format for YOLOX.s"""

    @staticmethod
    def convert(batch: List[AnnotatedFrame]) -> Dict[str, object]:
        """
        Converts a batch of annotated frames into an expected format for YOLOX.

        Args:
            batch (List[AnnotatedFrame]): the batch to convert

        Returns:
            Dict[str, object]: the converted batch
        """
        boxes, images, labels = YOLOXBatchConverter._get_images_boxes_labels(batch)

        return {
            "img": torch.tensor(np.stack(images), dtype=torch.uint8),
            "gt_boxes": boxes,
            "gt_classes": labels,
            "gt_num": [len(l) for l in labels]
        }

    @staticmethod
    def _get_images_boxes_labels(batch: List[AnnotatedFrame]) -> Tuple[List, List, List]:
        """Gets lists of converted images, bounding boxes and labels for the batch."""
        images = []
        boxes = []
        labels = []

        for annotated_frame in batch:
            img = annotated_frame.frame
            height, width = img.shape[:2]
            images.append(img)

            frame_boxes = []
            frame_labels = []

            for ann in annotated_frame.annotations:
                bbox_scaler = StaticBBoxScaler(height, width)
                scaled_bbox = bbox_scaler.scale(ann.bbox)

                bbox_converter = BBoxToCorners()
                frame_boxes.append(bbox_converter.convert(scaled_bbox))
                frame_labels.append(ann.cls.value)

            boxes.append(torch.tensor(frame_boxes, dtype=torch.float32))
            labels.append(torch.tensor(frame_labels, dtype=torch.int64))

        return boxes, images, labels