from dataclasses import dataclass
from typing import Tuple, List, Optional

import numpy as np

from src.typevars.enum_type import T_Enum


@dataclass(frozen=True)
class Instance:
    """Holds frame-annotation instance-related information in an immutable structure."""
    source: str
    index: int
    data: np.ndarray
    annotations: Optional[List[Tuple[T_Enum, float, float, float, float]]]
    end_of_stream: bool
