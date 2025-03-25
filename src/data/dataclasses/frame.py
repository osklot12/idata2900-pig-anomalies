from dataclasses import dataclass

import numpy as np

from src.data.dataclasses.source_metadata import SourceMetadata


@dataclass(frozen=True)
class Frame:
    """
    Represents a single video frame containing raw pixel data.

    Attributes:
        source (SourceMetadata): the source metadata
        index (int): the frame index within its source
        data (np.ndarray): the raw pixel data
        end_of_stream (bool): whether the frame is at the end of the stream
    """
    source: SourceMetadata
    index: int
    data: np.ndarray
    end_of_stream: bool