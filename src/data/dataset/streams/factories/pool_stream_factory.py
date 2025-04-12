from src.data.dataset.streams.factories.stream_factory import StreamFactory, T
from src.data.dataset.streams.pool_stream import PoolStream
from src.data.dataset.streams.stream import Stream


class PoolStreamFactory(StreamFactory):
    """Factory for creating PoolStream instances."""

    def __init__(self, pool_size: int):
        """
        Initializes a PoolStreamFactory instance.

        Args:
            pool_size (int): the size of the pool streams
        """
        self._pool_size = pool_size

    def create_stream(self) -> Stream[T]:
        return PoolStream(self._pool_size)