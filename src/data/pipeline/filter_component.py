from typing import TypeVar, Generic, Callable

from src.data.pipeline.base_component import BaseComponent
from src.data.pipeline.consumer import Consumer

# input type
I = TypeVar("I")

# output type
O = TypeVar("O")


class FilterComponent(Generic[I, O], BaseComponent):
    """Filters data based on a condition."""

    def __init__(self, condition: Callable[[I], bool]):
        """
        Initializes a FilterComponent instance.

        Args:
            condition (Callable[[I], bool]): condition that decides whether to keep the data or not
        """
        super().__init__()
        self._condition = condition

    def _consume(self, data: I, consumer: Consumer[O]) -> bool:
        success = True

        if self._condition(data):
            success = consumer.consume(data)

        return success