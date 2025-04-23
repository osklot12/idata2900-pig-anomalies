from typing import TypeVar

from src.data.dataset.streams.closable_stream import ClosableStream
from src.data.dataset.streams.factories.stream_factory import ClosableStreamFactory

# stream data type
T = TypeVar("T")

class ReusableStreamFactory(ClosableStreamFactory[T]):
    """Factory adapter for returning the same stream for each creation."""

    def __init__(self, stream: ClosableStream[T]):
        """
        Initializes a ReusableStreamFactory instance.

        Args:
            stream (ClosableStream[T]): stream to return
        """
        self._stream = stream

    def