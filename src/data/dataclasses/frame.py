from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Frame:
    """Represents a single video frame containing raw pixel data."""
    source: str
    index: int
    data: np.ndarray
    end_of_stream: bool