from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class SourceMetadata:
    """
    Stores metadata for a data source.

    Attributes:
        source_id (str): the ID of the data source
        frame_resolution (Tuple[int, int]): the resolution of the frames within the data source
    """
    source_id: str
    frame_resolution: Tuple[int, int]