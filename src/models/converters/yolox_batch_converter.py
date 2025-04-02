from typing import List, Dict, Tuple

import numpy as np
import torch

from src.data.dataclasses.annotated_frame import AnnotatedFrame
from src.models.converters.bbox_to_corners import BBoxToCorners


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
            images.append(annotated_frame.frame)

            frame_boxes = []
            frame_labels = []

            for ann in annotated_frame.annotations:
                bbox_converter = BBoxToCorners()
                frame_boxes.append(bbox_converter.convert(ann.bbox))
                frame_labels.append(ann.cls.value)

            boxes.append(torch.tensor(frame_boxes, dtype=torch.float32))
            labels.append(torch.tensor(frame_labels, dtype=torch.int64))

        return boxes, images, labels