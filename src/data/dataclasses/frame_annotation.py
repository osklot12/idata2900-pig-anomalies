from dataclasses import dataclass
from typing import List

from src.data.dataclasses.bbox_annotation import BBoxAnnotation


@dataclass(frozen=True)
class FrameAnnotation:
    """Represents annotations associated with a specific frame in a video stream."""
    source: str
    index: int
    annotations: [List[BBoxAnnotation]]
    end_of_stream: bool