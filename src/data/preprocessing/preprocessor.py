from abc import ABC, abstractmethod
from typing import TypeVar, Generic, List

T = TypeVar("T")


class Preprocessor(Generic[T], ABC):
    """Interface for preprocessors."""

    @abstractmethod
    def process(self, instance: T) -> List[T]:
        """
        Processes the given instance.

        Args:
            instance (T): the instance to process

        Returns:
            List[T]: processed instance(s)
        """
        raise NotImplementedError