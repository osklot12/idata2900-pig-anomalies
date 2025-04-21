from typing import TypeVar, Generic

from src.data.dataset.streams.dock_stream import DockStream
from src.data.dataset.streams.factories.writable_stream_factory import WritableStreamFactory
from src.data.dataset.streams.stream import Stream

T = TypeVar("T")

class DockStreamFactory(Generic[T], WritableStreamFactory[T]):
    """Factory for creating DockStream instances."""

    def __init__(self, buffer_size: int, dock_size: int):
        """
        Initializes a DockStreamFactory instance.

        Args:
            buffer_size (int): the size of the external queue size
            dock_size (int): the size of the internal queue size
        """
        self._buffer_size = buffer_size
        self._dock_size = dock_size

    def create_stream(self) -> Stream[T]:
        return DockStream(buffer_size=self._buffer_size, dock_size=self._dock_size)