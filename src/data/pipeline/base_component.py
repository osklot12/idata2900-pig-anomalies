from typing import TypeVar, Optional, Generic
from abc import ABC, abstractmethod

from src.data.pipeline.component import Component
from src.data.pipeline.consumer import Consumer
from src.data.structures.atomic_var import AtomicVar

# input type
I = TypeVar("I")

# output type
O = TypeVar("O")


class BaseComponent(Generic[I, O], Component[I, O], ABC):
    """Base class for pipeline components providing thread-safe connection to consumers."""

    def __init__(self, consumer: Optional[Consumer[O]] = None):
        """
        Initializes a BaseComponent instance.

        Args:
            consumer (Optional[Consumer[I]]): optional consumer of the data
        """
        self._consumer = AtomicVar[Consumer[O]](consumer)

    def consume(self, data: Optional[I]) -> bool:
        success = False

        consumer = self._consumer.get()
        if consumer is not None:
            if data is not None:
                success = self._consume(data, consumer)
            else:
                success = consumer.consume(data)

        return success

    @abstractmethod
    def _consume(self, data: I, consumer: Consumer[O]) -> bool:
        """Subclass implementation of consume, guarding against consumer set to None and None data."""
        raise NotImplementedError

    def connect(self, consumer: Consumer[O]) -> None:
        self._consumer.set(consumer)