import torch

def maybe_wrap_for_yolo(x):
    # Only wrap in tuple if model expects a list-like input
    if isinstance(x, torch.Tensor):
        return (x,)  # match YOLOv8 expected input shape
    return x

