import threading
from collections import deque
from typing import Dict, Generic, TypeVar, Optional

# generic type for stored data
T = TypeVar('T')

class IndexedBuffer(Generic[T]):
    """Thread-safe buffer that stores indexed data with automatic eviction."""

    def __init__(self, max_size: int = 1000):
        self.data: Dict[int, T] = {}
        self.order = deque()
        self.max_size = max_size
        self.lock = threading.Lock()

    def add(self, index: int, value: T):
        """Adds an indexed item and handles automatic eviction."""
        with self.lock:
            if index not in self.data:
                self.data[index] = value
                self.order.append(index)

                if len(self.order) > self.max_size:
                    old_index = self.order.popleft()
                    del self.data[old_index]

    def pop(self, index: int) -> Optional[T]:
        """Removes and returns an indexed item if it exists."""
        result = None

        with self.lock:
            if index in self.data:
                self.order.remove(index)
                result = self.data.pop(index)

        return result


    def has(self, index: int) -> bool:
        """Checks if an indexed item exists."""
        with self.lock:
            return index in self.data

    def size(self) -> int:
        """Returns the current number of stored items."""
        with self.lock:
            return len(self.data)