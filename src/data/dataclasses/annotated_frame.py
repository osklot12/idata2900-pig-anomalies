from dataclasses import dataclass
from typing import List

import numpy as np

from src.data.dataclasses.annotated_bbox import AnnotatedBBox


@dataclass(frozen=True)
class AnnotatedFrame:
    """Represents a single video frame along with its associated annotations."""
    frame: np.ndarray
    annotations: List[AnnotatedBBox]