from abc import abstractmethod

from typing import Callable, Optional

from src.data.dataclasses.annotation import Annotation
from src.data.loading.feed_status import FeedStatus
from src.data.streaming.streamers.streamer import Streamer
from src.data.streaming.streamers.streamer_status import StreamerStatus


class AnnotationStreamer(Streamer):
    """A streamer for streaming annotation data."""

    def __init__(self, callback: Callable[[Annotation], FeedStatus]):
        """
        Initializes a AnnotationStreamer instance.

        Args:
            callback (Callable[[Annotation], FeedStatus]): the callback to feed data
        """
        super().__init__()
        self._callback = callback

    def _stream(self) -> StreamerStatus:
        result = StreamerStatus.COMPLETED

        annotation = self._get_next_annotation()
        while annotation is not None and not self._is_requested_to_stop():
            self._callback(annotation)
            annotation = self._get_next_annotation()

        if annotation and self._is_requested_to_stop():
            result = StreamerStatus.STOPPED

        return result

    @abstractmethod
    def _get_next_annotation(self) -> Optional[Annotation]:
        """
        Returns the next annotation for the stream.

        Returns:
            Optional[Annotation]: the next annotation, or None if end of stream
        """
        raise NotImplementedError