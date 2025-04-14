from abc import abstractmethod
from typing import Optional

from src.data.dataclasses.frame import Frame
from src.data.pipeline.consumer import Consumer
from src.data.pipeline.producer import T
from src.data.streaming.streamers.concurrent_streamer import ConcurrentStreamer
from src.data.streaming.streamers.producer_streamer import ProducerStreamer
from src.data.streaming.streamers.streamer_status import StreamerStatus
from src.data.structures.atomic_var import AtomicVar

class VideoStreamer(ConcurrentStreamer, ProducerStreamer[Frame]):
    """A streamer for streaming video data."""

    def __init__(self, consumer: Optional[Consumer[Frame]] = None):
        """
        Initializes a VideoStreamer instance.

        Args:
            consumer (Optional[Consumer[Frame]]): optional consumer of the streamed data
        """
        super().__init__()
        self._consumer = AtomicVar[Consumer[Frame]](consumer)

    def _stream(self) -> StreamerStatus:
        result = StreamerStatus.COMPLETED

        frame = self._get_next_frame()
        while frame is not None and not self._is_requested_to_stop():
            consumer = self._consumer.get()
            if consumer is not None:
                consumer.consume(frame)
                frame = self._get_next_frame()

        # indicating end of streams
        consumer = self._consumer.get()
        if consumer is not None:
            self._consumer.get().consume(None)

        if frame and self._is_requested_to_stop():
            result = StreamerStatus.STOPPED

        return result

    @abstractmethod
    def _get_next_frame(self) -> Optional[Frame]:
        """
        Returns the next frame for the streams.

        Returns:
            Optional[Frame]: next Frame, or None if end of streams
        """
        raise NotImplementedError

    def connect(self, consumer: Consumer[T]) -> None:
        self._consumer.set(consumer)