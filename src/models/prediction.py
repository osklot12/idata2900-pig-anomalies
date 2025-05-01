from dataclasses import dataclass

@dataclass(frozen=True)
class Prediction:
    """
    An object detection prediction.

    Attributes:
        x1 (float): the min x coordinate for bounding box
        y1 (float): the min y coordinate for bounding box
        x2 (float): the max x coordinate for bounding box
        y2 (float): the max y coordinate for bounding box
        conf (float): the confidence score for the prediction
        cls (int):
    """
    x1: float
    y1: float
    x2: float
    y2: float
    conf: float
    cls: int