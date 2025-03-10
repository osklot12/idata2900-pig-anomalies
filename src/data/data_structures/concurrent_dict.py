import threading
from typing import TypeVar, Dict, Generic, List, Tuple

K = TypeVar("K")
V = TypeVar("V")

class ConcurrentDict(Generic[K, V]):
    """A thread-safe dictionary."""

    def __init__(self):
        """Initializes an instance of ConcurrentDict."""
        self._dict: Dict[K, V] = {}
        self._lock = threading.Lock()

    def set(self, key: K, value: V) -> None:
        """
        Sets a key-value pair.

        Args:
            key (K): The key to set.
            value (V): The value to set.
        """
        with self._lock:
            self._dict[key] = value

    def get(self, key: K) -> V:
        """
        Returns the value associated with key.

        Args:
            key (K): The key for which to get the value.
        """
        with self._lock:
            return self._dict[key]

    def remove(self, key: K) -> None:
        """
        Removes the value associated with key.
        """
        with self._lock:
            if key in self._dict:
                del self._dict[key]

    def contains(self, key: K) -> bool:
        """
        Indicates whether dict contains a certain key.

        Args:
            key (K): The key to check.
        """
        with self._lock:
            return key in self._dict

    def items(self) -> List[Tuple[K, V]]:
        """
        Returns a thread-safe copy of items.

        Returns:
            List[Tuple[K, V]]: A thread-safe copy of items.
        """
        with self._lock:
            return list(self._dict.items())

    def values(self) -> List[V]:
        """
        Returns a thread-safe copy of values.

        Returns:
            List[V]: A thread-safe copy of values.
        """
        with self._lock:
            return list(self._dict.values())

    def keys(self) -> List[K]:
        """
        Returns a thread-safe copy of keys.

        Returns:
            List[K]: A thread-safe copy of keys.
        """
        with self._lock:
            return list(self._dict.keys())

    def clear(self) -> None:
        """Clears the dictionary."""
        with self._lock:
            self._dict.clear()

    def __len__(self) -> int:
        """
        Returns the length of the dictionary.

        Returns:
            int: The length of the dictionary.
        """
        with self._lock:
            return len(self._dict)