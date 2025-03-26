from dataclasses import dataclass

from src.data.dataclasses.bbox import BBox
from src.typevars.enum_type import T_Enum


@dataclass(frozen=True)
class AnnotatedBBox:
    """
    Represents an object detected within a frame.

    Attributes:
        cls (T_Enum): the class label for the bounding box
        bbox (BBox): the bounding box
    """
    cls: T_Enum
    bbox: BBox