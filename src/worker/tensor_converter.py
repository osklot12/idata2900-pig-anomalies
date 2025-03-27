import torch
import numpy as np
from src.data.dataclasses.annotated_frame import AnnotatedFrame

def convert_to_tensor_format(annotated: AnnotatedFrame):
    """Converts AnnotatedFrame to Faster R-CNN compatible tensors."""
    frame = annotated.frame
    annotations = annotated.annotations

    image_tensor = torch.tensor(frame, dtype=torch.float32).permute(2, 0, 1) / 255.0

    # Extract bbox and label from AnnotatedBBox objects
    boxes = [
        [ann.bbox.x, ann.bbox.y, ann.bbox.x + ann.bbox.width, ann.bbox.y + ann.bbox.height]
        for ann in annotations
    ]

    labels = [ann.cls.value for ann in annotations]  # assumes cls is Enum or castable to int

    target = {
        "boxes": torch.tensor(boxes, dtype=torch.float32),
        "labels": torch.tensor(labels, dtype=torch.int64)
    }

    return image_tensor, target
