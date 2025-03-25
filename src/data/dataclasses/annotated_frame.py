from dataclasses import dataclass
from typing import List

import numpy as np

from src.data.dataclasses.annotated_bbox import AnnotatedBBox


@dataclass(frozen=True)
class AnnotatedFrame:
    """
    Represents a single video frame along with its associated annotations.

    Attributes:
        frame (np.ndarray): the frame data
        annotations (List[AnnotatedBBox]): the annotations for the frame
    """
    frame: np.ndarray
    annotations: List[AnnotatedBBox]