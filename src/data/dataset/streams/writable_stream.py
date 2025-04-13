from abc import abstractmethod
from typing import TypeVar, Generic

from src.data.dataset.streams.stream import Stream
from src.data.pipeline.consumer_provider import ConsumerProvider

T = TypeVar("T")


class WritableStream(Generic[T], Stream[T], ConsumerProvider[T]):
    """Interface for streams what can be written to."""

    @abstractmethod
    def close(self) -> None:
        """Closes the stream."""
        raise NotImplementedError