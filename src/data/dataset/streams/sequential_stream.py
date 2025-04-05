import queue
from queue import Queue
from typing import TypeVar, Generic, Optional

from src.data.dataset.streams.queue_stream import QueueStream
from src.data.dataset.streams.stream import Stream
from src.data.streaming.factories.aggregated_streamer_factory import AggregatedStreamerFactory

T = TypeVar("T")

class SequentialStream(Generic[T], Stream):
    """Sequential streams of data, order of data items is predetermined."""

    def __init__(self, streamer_factory: AggregatedStreamerFactory):
        """
        Initializes a SequentialStream instance.

        Args:
            streamer_factory (AggregatedStreamerFactory): the factory for creating aggregated streamers
        """
        self._streamer_factory = streamer_factory
        self._streamer_streams: Queue[Queue[T]] = queue.Queue()
        self._queue_stream = QueueStream[T]()

    def read(self) -> Optional[T]:
        pass

