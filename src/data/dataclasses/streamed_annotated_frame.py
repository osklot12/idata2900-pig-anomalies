from dataclasses import dataclass
from typing import List

import numpy as np

from src.data.dataclasses.annotated_bbox import AnnotatedBBox
from src.data.dataclasses.source_metadata import SourceMetadata


@dataclass(frozen=True)
class StreamedAnnotatedFrame:
    """
    Represents a single video frame along with its associated annotations and metadata.

    Attributes:
        source (SourceMetadata): the source metadata
        index (int): the frame index within its source
        frame (np.ndarray): the raw frame pixel data
        annotations (List[AnnotatedBBox]): the annotations associated with the frame
        end_of_stream (bool): whether the frame is at the end of the stream
    """
    source: SourceMetadata
    index: int
    frame: np.ndarray
    annotations: List[AnnotatedBBox]
    end_of_stream: bool
