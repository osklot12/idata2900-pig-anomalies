from dataclasses import dataclass
from typing import List

from src.data.dataclasses.annotated_bbox import AnnotatedBBox
from src.data.dataclasses.source_metadata import SourceMetadata


@dataclass(frozen=True)
class FrameAnnotations:
    """
    Represents annotations associated with a specific frame in a video stream.

    Attributes:
        source (SourceMetadata): the source metadata
        index (int): the frame index within its source
        annotations (List[AnnotatedBBox]): the annotations associated with the frame
        end_of_stream (bool): whether the frame is at the end of the stream
    """
    source: SourceMetadata
    index: int
    annotations: List[AnnotatedBBox]
    end_of_stream: bool