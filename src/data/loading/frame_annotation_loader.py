from typing import Callable, Tuple, Optional, List
import numpy as np

from src.data.loading.feed_status import FeedStatus
from src.utils.norsvin_annotations import NorsvinBehaviorClass


class FrameAnnotationLoader:
    """Loads pairs of frames and annotations and feeds them to a callback function."""

    def __init__(self, callback: Callable[[str, int, np.ndarray,
                                           Optional[List[Tuple[NorsvinBehaviorClass, float, float, float, float], bool]]], FeedStatus]):
        self.callback = callback

    def feed_frame(self, source: str, index: int, data: np.ndarray, end_of_stream: bool) -> FeedStatus:
        """Feeds a frame."""

    def feed_annotation(self, source: str, index: int,
                        annotations: Optional[List[Tuple[NorsvinBehaviorClass, float, float, float, float], bool]],
                        end_of_stream: bool) -> FeedStatus:
        """Feeds an annotation."""
