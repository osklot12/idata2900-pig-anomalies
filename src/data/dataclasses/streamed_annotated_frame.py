from dataclasses import dataclass
from typing import Tuple, List, Optional

import numpy as np

from src.data.dataclasses.annotated_bbox import AnnotatedBBox
from src.data.dataclasses.frame_annotations import FrameAnnotations
from src.typevars.enum_type import T_Enum


@dataclass(frozen=True)
class StreamedAnnotatedFrame:
    """Represents a single video frame along with its associated annotations and metadata."""
    source: str
    index: int
    frame: np.ndarray
    annotations: List[AnnotatedBBox]
    end_of_stream: bool
