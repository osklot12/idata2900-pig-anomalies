from abc import ABC, abstractmethod
from typing import TypeVar, Generic

from src.data.pipeline.component import Component

T = TypeVar("T")

class ComponentFactory(Generic[T], ABC):
    """Interface for component factories."""

    @abstractmethod
    def create_component(self) -> Component[T]:
        """
        Creates and returns a new Component instance.

        Returns:
            Component[T]: a new Component instance
        """
        raise NotImplementedError