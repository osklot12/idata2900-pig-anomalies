from dataclasses import dataclass
from typing import List, Tuple

from src.data.dataclasses.annotated_bbox import AnnotatedBBox
from src.data.dataclasses.source_metadata import SourceMetadata


@dataclass(frozen=True)
class CompressedAnnotatedFrame:
    """
    Represents a single compressed video frame along with its associated annotations and metadata.

    Attributes:
        source (SourceMetadata): the source metadata
        index (int): the frame index within its source
        frame (bytes): the compressed frame data
        shape (Tuple[int, int, int]): the shape of the frame
        dtype (str): the data type of the frame
        annotations (List[AnnotatedBBox]): the annotations associated with the frame
    """
    source: SourceMetadata
    index: int
    frame: bytes
    shape: Tuple[int, int, int]
    dtype: str
    annotations: List[AnnotatedBBox]