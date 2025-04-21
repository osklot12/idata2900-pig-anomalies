from abc import ABC
from typing import TypeVar, Generic

from src.data.dataset.streams.closable_stream import ClosableStream
from src.data.pipeline.consumer_provider import ConsumerProvider

# stream data type
T = TypeVar("T")


class WritableStream(Generic[T], ClosableStream[T], ConsumerProvider[T], ABC):
    """Interface for streams what can be written to."""