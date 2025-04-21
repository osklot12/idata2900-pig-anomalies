from abc import ABC, abstractmethod
from typing import TypeVar, Generic, Optional

from src.data.pipeline.consumer import Consumer
from src.data.structures.atomic_bool import AtomicBool

T = TypeVar("T")

class ConsumerProvider(Generic[T], ABC):
    """Interface for providers of consumers."""

    @abstractmethod
    def get_consumer(self, release: Optional[AtomicBool] = None) -> Optional[Consumer[T]]:
        """
        Returns a consumer that can consume data.

        Args:
            release (Optional[AtomicBool]): optional for releasing blocks

        Returns:
            Optional[Consumer[T]]: consumer to send data into, or None if no consumer is available
        """
        raise NotImplementedError