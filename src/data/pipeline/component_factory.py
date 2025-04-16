from abc import ABC, abstractmethod
from typing import TypeVar, Generic

from src.data.pipeline.component import Component

I = TypeVar("I")
O = TypeVar("O")

class ComponentFactory(Generic[I, O], ABC):
    """Interface for component factories."""

    @abstractmethod
    def create_component(self) -> Component[I, O]:
        """
        Creates and returns a new Component instance.

        Returns:
            Component[T]: a new Component instance
        """
        raise NotImplementedError