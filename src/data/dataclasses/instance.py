from dataclasses import dataclass
from typing import Tuple, List, Optional

import numpy as np

from src.data.dataclasses.bbox_annotation import BBoxAnnotation
from src.data.dataclasses.frame_annotation import FrameAnnotation
from src.typevars.enum_type import T_Enum


@dataclass(frozen=True)
class Instance:
    """Holds frame-annotation instance-related information in an immutable structure."""
    source: str
    index: int
    data: np.ndarray
    annotation: List[BBoxAnnotation]
    end_of_stream: bool
