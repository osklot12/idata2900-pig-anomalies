from abc import ABC
from typing import TypeVar, Generic

from src.data.dataset.streams.closable import Closable
from src.data.dataset.streams.stream import Stream
from src.data.pipeline.consumer_provider import ConsumerProvider

T = TypeVar("T")


class WritableStream(Generic[T], Stream[T], ConsumerProvider[T], Closable, ABC):
    """Interface for streams what can be written to."""