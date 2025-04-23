from typing import TypeVar, Generic

from src.data.dataset.streams.closable_stream import ClosableStream
from src.data.dataset.streams.providers.stream_provider import StreamProvider
from src.data.dataset.streams.stream import Stream

# stream data type
T = TypeVar("T")


class ReusableStreamProvider(Generic[T], StreamProvider[T]):
    """Provider of a reusable stream, returning the same stream for every request."""

    def __init__(self, stream: ClosableStream[T]):
        """
        Initializes a ReusableStreamProvider instance.

        Args:
            stream (ClosableStream[T]): stream to return
        """
        self._stream = stream

    def get_stream(self) -> Stream[T]:
        return self._stream