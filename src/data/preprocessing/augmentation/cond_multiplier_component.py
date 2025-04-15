import copy
from typing import TypeVar, Generic, Optional, Callable

from src.data.pipeline.component import Component
from src.data.pipeline.consumer import Consumer
from src.data.structures.atomic_var import AtomicVar

T = TypeVar("T")

class CondMultiplierComponent(Generic[T], Component[T]):
    """Component that multiplies data based on some condition."""

    def __init__(self, n: int, predicate: Callable[[T], bool]):
        """
        Initializes a CondMultiplierComponent instance.

        Args:
            n (int): the multiplication factor
            predicate (Callable[[T], bool]): predicate that decides if the instance will be multiplied
        """
        self._n = n
        self._predicate = predicate
        self._consumer = AtomicVar[Consumer[T]](None)

    def consume(self, data: Optional[T]) -> bool:
        success = True

        consumer = self._consumer.get()
        if consumer is not None:
            i = 0
            m = self._n if (data is not None and self._predicate(data)) else 1
            while i < m and success:
                success = consumer.consume(copy.deepcopy(data))
                i += 1

        return success

    def connect(self, consumer: Consumer[T]) -> None:
        self._consumer.set(consumer)