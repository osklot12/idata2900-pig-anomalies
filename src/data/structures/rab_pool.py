import threading
import time
import random
from typing import TypeVar, Generic, Optional

from src.data.structures.atomic_bool import AtomicBool
from src.data.structures.atomic_var import AtomicVar

T = TypeVar("T")

WAITING_TIMEOUT = 0.1


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
        self._maxsize = AtomicVar[int](maxsize)
        self._min_ready = AtomicVar[int](min_ready)

        self._lock = threading.Lock()
        self._not_full = threading.Condition(self._lock)
        self._ready = threading.Condition(self._lock)

    def put(self, item: T, release: Optional[AtomicBool] = None) -> bool:
        """
        Puts an item into the pool.

        Args:
            item (T): the item to put
            release (Optional[AtomicBool]): optional release for unblocking the operation
        """
        success = False

        if item is not None:
            print(f"[RABPool] Putting item: {type(item)}")
            released = False
            with self._not_full:
                while len(self._items) >= self._maxsize.get() and not released:
                    released = self._is_released(release)
                    self._not_full.wait(WAITING_TIMEOUT)

                if not released:
                    self._items.append(item)
                    success = True
                self._ready.notify()

        print(f"[RABPool] Pool size: {len(self)}")
        return success

    def get(self, release: Optional[AtomicBool] = None) -> Optional[T]:
        """
        Gets a random item from the pool.

        Args:
            release (Optional[AtomicBool]): optional release for unblocking the operation

        Returns:
            Optional[T]: the random item, or None if released
        """
        item = None

        released = False
        with self._ready:
            while len(self._items) < self._min_ready.get() and not released:
                released = self._is_released(release)
                self._ready.wait(WAITING_TIMEOUT)

            if not released and len(self._items) > 0:
                index = random.randint(0, len(self._items) - 1)
                item = self._items.pop(index)
            self._not_full.notify()

        return item

    @staticmethod
    def _is_released(release: Optional[AtomicBool]) -> bool:
        """Checks whether the release is released."""
        return release is not None and release

    def is_full(self) -> bool:
        """
        Returns whether the pool is full.

        Returns:
            bool: True if pool is full, False otherwise
        """
        with self._lock:
            return len(self._items) >= self._maxsize.get()

    def __len__(self) -> int:
        with self._lock:
            return len(self._items)
