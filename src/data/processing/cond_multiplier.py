import copy
from typing import TypeVar, List, Callable

from src.data.processing.processor import Processor, I, O

T = TypeVar("T")


class CondMultiplier(Processor[T, List[T]]):
    """Multiplies objects based on some condition."""

    def __init__(self, n: int, condition: Callable[[T], bool]):
        """
        Initializes a CondMultiplier instance.

        Args:
            n (int): multiplication factor
            condition (Callable[[T], bool]): condition that decides whether to multiply the instance
        """
        self._n = n
        self._condition = condition

    def process(self, data: I) -> List[T]:
        outputs = []

        if self._condition(data):
            for _ in range(self._n):
                outputs.append(copy.deepcopy(data))
        else:
            outputs.append(data)

        return outputs