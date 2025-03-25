from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class SourceMetadata:
    """Stores metadata for a data source."""
    source_id: str
    frame_resolution: Tuple[int, int]