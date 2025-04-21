from abc import ABC
from typing import TypeVar, Generic

from src.data.dataset.streams.closable import Closable
from src.data.dataset.streams.stream import Stream

# stream data type
T = TypeVar("T")


class ClosableStream(Generic[T], Stream[T], Closable, ABC):
    """Interface for closable streams."""