from typing import Generic, TypeVar, Optional

from src.data.pipeline.consumer import Consumer
from src.data.structures.atomic_bool import AtomicBool
from src.data.structures.rab_pool import RABPool

T = TypeVar("T")

CONSUME_LOOP_TIMEOUT = 0.1

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
        success = False

        keep_trying = True
        while keep_trying and not success:
            success = self._pool.put(item=data, timeout=CONSUME_LOOP_TIMEOUT)

            if self._release is not None and self._release:
                keep_trying = False

        return success