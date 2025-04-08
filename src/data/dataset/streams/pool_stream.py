from typing import TypeVar, Optional

from src.data.dataset.streams.stream import Stream

T = TypeVar("T")

class PoolStream(Stream):
    """Stream reading randomly from a pool of instances."""

    def read(self) -> Optional[T]:
        pass