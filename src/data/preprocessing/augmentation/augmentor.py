from abc import ABC, abstractmethod
from typing import TypeVar, Generic, List

T = TypeVar("T")


class Augmentor(Generic[T], ABC):
    """Interface for augmentor classes."""

    @abstractmethod
    def augment(self, instance: T) -> List[T]:
        """
        Augments the given instance.

        Args:
            instance (T): the instance to augment

        Returns:
            List[T]: the augmented instance(s)
        """
        raise NotImplementedError