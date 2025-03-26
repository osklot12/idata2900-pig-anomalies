import torch
import numpy as np

def convert_to_tensor_format(frame: np.ndarray, annotations: list):
    """Converts raw frame + annotations to Faster R-CNN input format."""
    image_tensor = torch.tensor(frame, dtype=torch.float32).permute(2, 0, 1) / 255.0

    boxes = [ann["bbox"] for ann in annotations]
    labels = [ann["label"] for ann in annotations]

    target = {
        "boxes": torch.tensor(boxes, dtype=torch.float32),
        "labels": torch.tensor(labels, dtype=torch.int64)
    }

    return image_tensor, target
