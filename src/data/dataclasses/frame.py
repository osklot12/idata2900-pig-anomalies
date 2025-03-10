from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Frame:
    """Holds frame-related information in an immutable structure."""
    source: str
    index: int
    data: np.ndarray
    end_of_stream: bool