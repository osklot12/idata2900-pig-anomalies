from typing import TypeVar, Generic, Callable

from src.data.pipeline.component import Component
from src.data.pipeline.component_factory import ComponentFactory
from src.data.preprocessing.augmentation.cond_multiplier_component import CondMultiplierComponent

T = TypeVar("T")

class CondMultiplierComponentFactory(Generic[T], ComponentFactory[T, T]):
    """Factory for creating CondMultiplierComponent instances."""

    def __init__(self, n: int, predicate: Callable[[T], bool]):
        """
        Initializes a CondMultiplierComponentFactory instance.

        Args:
            n (int): the multiplication factor
            predicate (Callable[[T], bool]): predicate that decides if instance will be multiplied or not
        """
        self._n = n
        self._predicate = predicate

    def create_component(self) -> Component[T, T]:
        return CondMultiplierComponent(n=self._n, predicate=self._predicate)