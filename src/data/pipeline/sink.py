import queue
from typing import TypeVar, Generic, Optional

from src.data.pipeline.consumer import Consumer

# data type to consume and return
T = TypeVar("T")

class Sink(Generic[T], Consumer[T]):
    """Simple endpoint for pipelines."""

    def __init__(self):
        """Initializes a Sink instance."""
        self._queue = queue.Queue()

    def consume(self, data: Optional[T]) -> bool:
        self._queue.put(data)
        return True

    def get(self) -> Optional[T]:
        """
        Returns the received data.

        Args:
            Optional[T]: the received data
        """
        data = None

        try:
            data = self._queue.get_nowait()

        except queue.Empty:
            pass

        return data

    def is_empty(self) -> bool:
        """
        Indicates if the sink is empty.

        Returns:
            bool: True if the sink is empty, otherwise False
        """
        return self._queue.empty()