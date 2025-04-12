from abc import ABC, abstractmethod
from typing import TypeVar, Generic

T = TypeVar("T")

class RuntimeParam(Generic[T], ABC):
    """Interface for values that are resolved at runtime."""

    @abstractmethod
    def resolve(self) -> T:
        """
        Returns the value.

        Returns:
            T: value
        """
        raise NotImplementedError