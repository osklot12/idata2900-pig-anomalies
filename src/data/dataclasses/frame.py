from dataclasses import dataclass

import numpy as np

from src.data.dataclasses.source_metadata import SourceMetadata


@dataclass(frozen=True)
class Frame:
    """Represents a single video frame containing raw pixel data."""
    source: SourceMetadata
    index: int
    data: np.ndarray
    end_of_stream: bool