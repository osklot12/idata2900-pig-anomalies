from typing import Generic, TypeVar, Optional

from src.data.dataset.streams.closable_stream import ClosableStream
from src.data.dataset.streams.factories.stream_factory import ClosableStreamFactory
from src.data.dataset.streams.providers.stream_provider import StreamProvider
from src.data.dataset.streams.stream import Stream

# stream data type
T = TypeVar("T")


class ClosingStreamProvider(Generic[T], StreamProvider[T]):
    """Provider of streams that closes the previous returned stream when another one is opened."""

    def __init__(self, stream_factory: ClosableStreamFactory[T]):
        """
        Initializes a ClosingStreamProvider instance.

        Args:
            stream_factory (ClosableStreamFactory[T]): factory for creating streams
        """
        self._stream_factory = stream_factory
        self._current_stream: Optional[ClosableStream[T]] = None

    def get_stream(self) -> Stream[T]:
        if self._current_stream is not None:
            self._current_stream.close()

        self._current_stream = self._stream_factory.create_stream()
        return self._current_stream