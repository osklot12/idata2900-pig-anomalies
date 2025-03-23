import threading
from collections import deque
from typing import Dict, Generic, TypeVar, Optional, Hashable, List

# generic type for stored data
T = TypeVar('T')


class HashBuffer(Generic[T]):
    """Thread-safe buffer that stores mapped data with automatic eviction."""

    def __init__(self, max_size: int = 1000):
        """
        Initializes a HashBuffer instance.

        Args:
            max_size (int, optional): the maximum size of the buffer, defaults to 1000
        """
        self._data: Dict[Hashable, T] = {}
        self._order = deque()
        self._max_size = max_size
        self._lock = threading.Lock()

    def add(self, key: Hashable, value: T):
        """
        Adds an item with any hashable key, handling automatic eviction.

        Args:
            key (Hashable): the hashable key
            value (T): the value to store
        """
        with self._lock:
            if key not in self._data:
                self._data[key] = value
                self._order.append(key)

                if len(self._order) > self._max_size:
                    old_index = self._order.popleft()
                    del self._data[old_index]

    def pop(self, key: Hashable) -> Optional[T]:
        """
        Removes and returns an item if it exists.

        Args:
            key (Hashable): the hashable key

        Returns:
            T: the item if it exists
        """
        result = None

        with self._lock:
            if key in self._data:
                self._order.remove(key)
                result = self._data.pop(key)

        return result

    def has(self, key: Hashable) -> bool:
        """
        Checks if an item exists.

        Returns:
            bool: true if the item exists, false otherwise
        """
        with self._lock:
            return key in self._data

    def at(self, key: Hashable) -> Optional[T]:
        """
        Thread-safe retrieval without removing the item.

        Args:
            key (Hashable): the hashable key

        Returns:
            T: the item if it exists, None otherwise
        """
        with self._lock:
            return self._data.get(key, None)

    def keys(self) -> List[Hashable]:
        """
        Thread-safe retrieval of all keys.

        Returns:
            List[Hashable]: the keys
        """
        with self._lock:
            return list(self._data.keys())

    def size(self) -> int:
        """
        Returns the current number of stored items.

        Returns:
            int: the current number of stored items
        """
        with self._lock:
            return len(self._data)
