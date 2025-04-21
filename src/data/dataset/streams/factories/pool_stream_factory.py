from typing import TypeVar, Generic

from src.data.dataset.streams.factories.writable_stream_factory import WritableStreamFactory
from src.data.dataset.streams.pool_stream import PoolStream
from src.data.dataset.streams.stream import Stream

T = TypeVar("T")

class PoolStreamFactory(Generic[T], WritableStreamFactory[T]):
    """Factory for creating PoolStream instances."""

    def __init__(self, pool_size: int, min_ready: int):
        """
        Initializes a PoolStreamFactory instance.

        Args:
            pool_size (int): the size of the pool
            min_ready (int): the minimum size of the pool before reading is allowed
        """
        self._pool_size = pool_size
        self._min_ready = min_ready

    def create_stream(self) -> Stream[T]:
        return PoolStream(pool_size=self._pool_size, min_ready=self._min_ready)