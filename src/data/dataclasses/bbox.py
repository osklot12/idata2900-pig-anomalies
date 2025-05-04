from dataclasses import dataclass

@dataclass(frozen=True)
class BBox:
    """
    Defines the spatial location of an object within a frame.

    Attributes:
        x (float): the x-coordinate of the top-left corner of the bounding box
        y (float): the y-coordinate of the top-left corner of the bounding box
        width (float): the width of the bounding box
        height (float): the height of the bounding box
    """
    x: float
    y: float
    width: float
    height: float