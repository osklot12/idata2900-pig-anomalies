from dataclasses import dataclass

@dataclass(frozen=True)
class BoundingBox:
    """Defines the spatial location of an object within a frame."""
    center_x: float
    center_y: float
    width: float
    height: float