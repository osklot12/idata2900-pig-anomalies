from dataclasses import dataclass

from src.data.dataclasses.bbox import BBox
from src.typevars.enum_type import T_Enum


@dataclass(frozen=True)
class AnnotatedBBox:
    """Represents an object detected within a frame."""
    cls: T_Enum
    bbox: BBox