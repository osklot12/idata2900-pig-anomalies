from abc import ABC, abstractmethod
from typing import TypeVar, Generic

from src.data.pipeline.consumer import Consumer

T = TypeVar("T")

class Producer(Generic[T], ABC):
    """Interface for producers of data."""

    @abstractmethod
    def connect(self, consumer: Consumer[T]) -> None:
        """
        Sets the consumer for the producer.

        Args:
            consumer (Consumer[T]): the consumer to set
        """
        raise NotImplementedError