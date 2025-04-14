from typing import Generic, TypeVar, Optional

from src.data.pipeline.consumer import Consumer
from src.data.structures.atomic_bool import AtomicBool
from src.data.structures.rab_pool import RABPool

T = TypeVar("T")


class ConsumingPool(Generic[T], Consumer[T]):
    """Consumer adapter for RABPool instances."""

    def __init__(self, pool: RABPool[T], release: Optional[AtomicBool] = None):
        """
        Initializes a ConsumingPool instance.

        Args:
            pool (RABPool[T]): the pool
            release (Optional[AtomicBool]): optional flag for releasing blocking behavior
        """
        self._pool = pool
        self._release = release

    def consume(self, data: Optional[T]) -> bool:
        print(f"[BBoxNormalizerComponent] Consumed instance and forward it to {self._pool}")
        return self._pool.put(data, self._release)