from dataclasses import dataclass
from typing import Optional, List

from src.data.dataclasses.annotation import Annotation


@dataclass(frozen=True)
class FrameAnnotation:
    """Represents annotations associated with a specific frame in a video stream."""
    source: str
    index: int
    annotations: Optional[List[Annotation]]
    end_of_stream: bool
