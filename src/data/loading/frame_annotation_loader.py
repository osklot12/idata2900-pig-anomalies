from typing import Callable, Tuple, Optional, List
import numpy as np

from src.data.data_structures.indexed_buffer import IndexedBuffer
from src.data.loading.feed_status import FeedStatus
from src.utils.norsvin_behavior_class import NorsvinBehaviorClass


class FrameAnnotationLoader:
    """Loads pairs of frames and annotations and feeds them to a callback function."""

    def __init__(self, callback: Callable[[str, int, np.ndarray,
                                           Optional[List[Tuple[NorsvinBehaviorClass, float, float, float, float]]], bool], FeedStatus],
                 buffer_size: int = 1000):
        self.callback = callback

        # frame buffer
        self.frame_buffer = IndexedBuffer[
            Tuple[str, np.ndarray, bool]
        ](max_size=buffer_size)

        # annotation buffer
        self.annotation_buffer = IndexedBuffer[
            Tuple[str, Optional[List[Tuple[NorsvinBehaviorClass, float, float, float, float]]], bool]
        ](max_size=buffer_size)

    def feed_frame(self, source: str, index: int, frame_data: np.ndarray, end_of_stream: bool) -> FeedStatus:
        """Feeds a frame."""
        result = FeedStatus.DROP

        if end_of_stream:
            index = -1

        if self.annotation_buffer.has(index):
            # match found
            annotations, annotation_end = self.annotation_buffer.pop(index)[1:]
            self._feed_pair(source, index, frame_data, annotations, end_of_stream)
            result = FeedStatus.ACCEPT

        else:
            # store frame until annotation arrives
            self.frame_buffer.add(index, (source, frame_data, end_of_stream))
            result = FeedStatus.ACCEPT

        return result

    def feed_annotation(self, source: str, index: int,
                        annotations: Optional[List[Tuple[NorsvinBehaviorClass, float, float, float, float]]],
                        end_of_stream: bool) -> FeedStatus:
        """Feeds an annotation and checks if a frame exists for this index."""
        result = FeedStatus.DROP

        if end_of_stream:
            index = -1

        if self.frame_buffer.has(index):
            # match found
            frame_data, frame_end = self.frame_buffer.pop(index)[1:]
            self._feed_pair(source, index, frame_data, annotations, end_of_stream)
            result = FeedStatus.ACCEPT

        else:
            # store annotation until frame arrives
            self.annotation_buffer.add(index, (source, annotations, end_of_stream))
            result = FeedStatus.ACCEPT

        return result

    def _feed_pair(self, source: str, index: int, frame_data: np.ndarray,
                   annotations: Optional[List[Tuple[NorsvinBehaviorClass, float, float, float, float]]],
                   end_of_stream: bool) -> FeedStatus:
        """Feeds matched frame and annotation pair to the callback function."""
        feed_result = FeedStatus.RETRY_LATER

        keep_feeding = True
        # keep feeding until fed or rejected
        while keep_feeding:
            feed_result = self.callback(source, index, frame_data, annotations, end_of_stream)
            if feed_result != FeedStatus.RETRY_LATER:
                keep_feeding = False

        return feed_result