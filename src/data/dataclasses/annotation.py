from dataclasses import dataclass
from typing import Optional, List, Tuple, TypeVar

from src.typevars.enum_type import T_Enum


@dataclass(frozen=True)
class Annotation:
    """Holds annotation-related information in an immutable structure."""
    source: str
    index: int
    annotations: Optional[List[Tuple[T_Enum, float, float, float, float]]]
    end_of_stream: bool
