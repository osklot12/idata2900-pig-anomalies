import threading
from collections import deque
from collections.abc import Sequence
from typing import Dict, Generic, TypeVar, Optional, Hashable, List

# generic type for stored data
K = TypeVar('K', bound=Hashable)
T = TypeVar('T')


class HashBuffer(Generic[K, T], Sequence):
    """Thread-safe buffer that stores mapped data with automatic eviction."""

    def __init__(self, max_size: int = 1000):
        """
        Initializes a HashBuffer instance.

        Args:
            max_size (int, optional): the maximum size of the buffer, defaults to 1000
        """
        self._data: Dict[K, T] = {}
        self._order = deque()
        self._max_size = max_size
        self._lock = threading.Lock()

    def add(self, key: K, value: T) -> List[K]:
        """
        Adds an item with any hashable key, handling automatic eviction.

        Args:
            key (K): the hashable key
            value (T): the value to store

        Returns:
            List[K]: list of evicted keys
        """
        evicted_keys = []

        with self._lock:
            if key in self._data:
                self._data[key] = value
            else:
                self._data[key] = value
                self._order.append(key)

                if len(self._order) > self._max_size:
                    old_index = self._order.popleft()
                    del self._data[old_index]
                    evicted_keys.append(old_index)

        return evicted_keys

    def pop(self, key: K) -> Optional[T]:
        """
        Removes and returns an item if it exists.

        Args:
            key (K): the hashable key

        Returns:
            T: the item if it exists
        """
        result = None

        with self._lock:
            if key in self._data:
                self._order.remove(key)
                result = self._data.pop(key)

        return result

    def has(self, key: K) -> bool:
        """
        Checks if an item exists.

        Returns:
            bool: true if the item exists, false otherwise
        """
        with self._lock:
            return key in self._data

    def at(self, key: K) -> Optional[T]:
        """
        Thread-safe retrieval without removing the item.

        Args:
            key (K): the hashable key

        Returns:
            T: the item if it exists, None otherwise
        """
        with self._lock:
            return self._data.get(key, None)

    def keys(self) -> List[K]:
        """
        Thread-safe retrieval of all keys.

        Returns:
            List[K]: the keys
        """
        with self._lock:
            return list(self._data.keys())

    def __contains__(self, key: K) -> bool:
        with self._lock:
            return key in self._data

    def __getitem__(self, index: int) -> T:
        with self._lock:
            key = list(self._order)[index]
            return self._data[key]

    def __len__(self) -> int:
        with self._lock:
            return len(self._data)

    def __iter__(self):
        with self._lock:
            for key in list(self._order):
                yield key, self._data[key]

    def __repr__(self) -> str:
        with self._lock:
            items = [(key, self._data[key]) for key in self._order]
            return f"{self.__class__.__name__}({items})"