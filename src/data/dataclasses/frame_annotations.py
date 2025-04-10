from dataclasses import dataclass
from typing import List

from src.data.dataclasses.annotated_bbox import AnnotatedBBox
from src.data.dataclasses.source_metadata import SourceMetadata


@dataclass(frozen=True)
class FrameAnnotations:
    """
    Represents annotations associated with a specific frame in a video streams.

    Attributes:
        source (SourceMetadata): the source metadata
        index (int): the frame index within its source
        annotations (List[AnnotatedBBox]): the annotations associated with the frame
    """
    source: SourceMetadata
    index: int
    annotations: List[AnnotatedBBox]