from typing import Callable, Tuple
import numpy as np

from src.data.loading.feed_status import FeedStatus
from src.utils.norsvin_annotations import NorsvinBehaviorClass


class FrameAnnotationLoader:
    """Loads pairs of frames and annotations and feeds them to a callback function."""

    def __init__(self, callback: Callable[[str, int, np.ndarray, Tuple[NorsvinBehaviorClass, float, float, float, float], bool], FeedStatus]):
        self.callback = callback