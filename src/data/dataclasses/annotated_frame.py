from dataclasses import dataclass
from typing import List

import numpy as np

from src.data.dataclasses.annotated_bbox import AnnotatedBBox
from src.data.dataclasses.identifiable import Identifiable
from src.data.dataclasses.source_metadata import SourceMetadata


@dataclass(frozen=True)
class AnnotatedFrame(Identifiable):
    """
    Represents a single video frame along with its associated annotations and metadata.

    Attributes:
        source (SourceMetadata): the source metadata
        index (int): the frame index within its source
        frame (np.ndarray): the raw frame pixel data
        annotations (List[AnnotatedBBox]): the annotations associated with the frame
    """
    source: SourceMetadata
    index: int
    frame: np.ndarray
    annotations: List[AnnotatedBBox]

    def get_id(self) -> str:
        return self.source.source_id
