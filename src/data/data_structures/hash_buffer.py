import threading
from collections import deque
from typing import Dict, Generic, TypeVar, Optional, Hashable, KeysView, List

# generic type for stored data
T = TypeVar('T')

class HashBuffer(Generic[T]):
    """Thread-safe buffer that stores mapped data with automatic eviction."""

    def __init__(self, max_size: int = 1000):
        self.data: Dict[Hashable, T] = {}
        self.order = deque()
        self.max_size = max_size
        self.lock = threading.Lock()

    def add(self, key: Hashable, value: T):
        """Adds an item with any hashable key, handling automatic eviction."""
        with self.lock:
            if key not in self.data:
                self.data[key] = value
                self.order.append(key)

                if len(self.order) > self.max_size:
                    old_index = self.order.popleft()
                    del self.data[old_index]

    def pop(self, key: Hashable) -> Optional[T]:
        """Removes and returns an item if it exists."""
        result = None

        with self.lock:
            if key in self.data:
                self.order.remove(key)
                result = self.data.pop(key)

        return result


    def has(self, key: Hashable) -> bool:
        """Checks if an item exists."""
        with self.lock:
            return key in self.data

    def at(self, key: Hashable) -> Optional[T]:
        """Thread-safe retrieval without removing the item."""
        with self.lock:
            return self.data.get(key, None)

    def keys(self) -> List[Hashable]:
        """Thread-safe retrieval of all keys."""
        with self.lock:
            return list(self.data.keys())

    def size(self) -> int:
        """Returns the current number of stored items."""
        with self.lock:
            return len(self.data)