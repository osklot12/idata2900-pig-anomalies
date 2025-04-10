from typing import TypeVar, Generic, Callable, List, Optional

from src.data.preprocessing.preprocessor import Preprocessor

T = TypeVar("T")


class CondMultiplier(Generic[T], Preprocessor[T]):
    """Multiplies instances whenever a predicate is true."""

    def __init__(self, n: int, predicate: Callable[[T], bool]):
        """
        Initializes a CondMultiplier instance.

        Args:
            n (int): number of instances to output
            predicate (Callable[[T], bool]): the predicate that decides whether to multiply
        """
        self._n = n
        self._predicate = predicate

    def process(self, instance: T) -> List[T]:
        result = [instance]

        if self._predicate(instance):
            for _ in range(self._n - 1):
                result.append(instance)

        return result