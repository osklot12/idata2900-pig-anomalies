from abc import ABC, abstractmethod
from typing import TypeVar, Generic

from src.data.preprocessing.augmentation.augmentors.augmentor import Augmentor

T = TypeVar("T")

class AugmentorFactory(Generic[T], ABC):
    """Interface for factories of Augmentor instances."""

    @abstractmethod
    def create_augmentor(self) -> Augmentor[T]:
        """
        Creates and returns an Augmentor instance.

        Returns:
            Augmentor[T]: the Augmentor instance
        """
        raise NotImplementedError