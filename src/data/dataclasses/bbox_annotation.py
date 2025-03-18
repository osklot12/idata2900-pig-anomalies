from dataclasses import dataclass

from src.data.dataclasses.bounding_box import BoundingBox
from src.typevars.enum_type import T_Enum


@dataclass(frozen=True)
class BBoxAnnotation:
    """Represents an object detected within a frame."""
    cls: T_Enum
    bbox: BoundingBox