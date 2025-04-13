import threading
import time
import random
from typing import TypeVar, Generic, Optional

T = TypeVar("T")

class RABPool(Generic[T]):
    """Thread-safe random access pool with blocking put and get methods (Random Access Blocking Pool)."""

    def __init__(self, maxsize: int = 3000, min_ready: int = 0):
        """
        Initializes a RandomAccessBlockingPool instance.

        Args:
            maxsize (int): the max size of the pool
            min_ready (int): the required min size of the pool to get instances
        """
        self._items: list[T] = []
        self._maxsize = maxsize
        self._min_ready = min_ready

        self._lock = threading.Lock()
        self._not_full = threading.Condition(self._lock)
        self._not_ready = threading.Condition(self._lock)

    def put(self, item: Optional[T], timeout: Optional[float] = None) -> bool:
        """
        Puts an item into the pool.

        Args:
            item (Optional[T]): the item to put
            timeout (Optional[float]): the time to wait before timing out
        """
        result = False

        with self._not_full:
            timed_out = False
            end_time = time.time() + timeout if timeout is not None else None
            while len(self._items) >= self._maxsize and not timed_out:
                remaining = end_time - time.time() if end_time is not None else None
                if remaining is not None and remaining <= 0:
                    timed_out = True
                self._not_full.wait(timeout=remaining)

            if not timed_out:
                self._items.append(item)
                result = True

            self._not_ready.notify()

        return result


    def get(self, timeout: Optional[float] = None) -> Optional[T]:
        """
        Gets a random item from the pool.

        Args:
            timeout (Optional[float]): the time to wait before timing out

        Returns:
            Optional[T]: the random item
        """
        result = None

        with self._not_ready:
            timed_out = False
            end_time = time.time() + timeout if timeout is not None else None
            while len(self._items) < self._min_ready and not timed_out:
                remaining = end_time - time.time() if end_time is not None else None
                if remaining is not None and remaining <= 0:
                    timed_out = True
                self._not_ready.wait(timeout=remaining)

            if not timed_out and self._items:
                index = random.randint(0, len(self._items) - 1)
                result = self._items.pop(index)
                self._not_full.notify()

        return result

    def is_full(self) -> bool:
        """
        Returns whether the pool is full.

        Returns:
            bool: True if pool is full, False otherwise
        """
        with self._lock:
            return len(self._items) >= self._maxsize

    def __len__(self) -> int:
        with self._lock:
            return len(self._items)