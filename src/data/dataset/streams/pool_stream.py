from collections import deque
from typing import TypeVar, Optional, Generic, Deque

from src.data.dataclasses.compressed_annotated_frame import CompressedAnnotatedFrame
from src.data.dataset.streams.writable_stream import WritableStream
from src.data.pipeline.consumer import Consumer
from src.data.pipeline.consuming_pool import ConsumingPool
from src.data.structures.atomic_bool import AtomicBool
from src.data.structures.rab_pool import RABPool

# stream data type
T = TypeVar("T")

# telemetry


class PoolStream(Generic[T], WritableStream[T]):
    """Stream reading randomly from a pool of instances."""

    def __init__(self, pool_size: int = 3000, min_ready: int = 2000):
        """
        Initializes a PoolStream instance.


        Args:
            pool_size (int): the max number of instances in the pool at a time, 3000 by default
            min_ready (int): the minimum number of instances before allowing reading
        """
        self._pool = RABPool[T](maxsize=pool_size, min_ready=min_ready)

        self._closed = AtomicBool(False)

    def read(self) -> Optional[T]:
        instance = None

        if not self._closed:
            instance = self._pool.get()

        return instance

    def get_consumer(self, release: Optional[AtomicBool] = None) -> Optional[Consumer[T]]:
        entry = None

        if not self._closed:
            entry = ConsumingPool[T](pool=self._pool, release=release)

        return entry

    def close(self) -> None:
        self._closed.set(True)