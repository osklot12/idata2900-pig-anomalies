from abc import ABC, abstractmethod
from typing import Generic, TypeVar

T = TypeVar("T")

class Augmentor(Generic[T], ABC):
    """Interface for classes augmenting data."""

    @abstractmethod
    def augment(self, data: T) -> T:
        """
        Augments the given data.

        Args:
            data (T): the data to augment

        Returns:
            T: the augmented data
        """
        raise NotImplementedError