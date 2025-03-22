from dataclasses import dataclass
from typing import List

from src.data.dataclasses.annotated_bbox import AnnotatedBBox


@dataclass(frozen=True)
class FrameAnnotations:
    """Represents annotations associated with a specific frame in a video stream."""
    source: str
    index: int
    annotations: List[AnnotatedBBox]
    end_of_stream: bool