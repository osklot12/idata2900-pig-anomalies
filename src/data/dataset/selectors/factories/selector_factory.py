from abc import ABC, abstractmethod
from typing import TypeVar, Generic, List

from src.data.dataset.selectors.selector import Selector

T = TypeVar("T")

class SelectorFactory(Generic[T], ABC):
    """Interface for Selector factories."""

    @abstractmethod
    def create_selector(self, candidates: List[T]) -> Selector[T]:
        """
        Creates and returns a Selector instance.

        Args:
            candidates (List[T]): list of the candidates to select from

        Returns:
            Selector[T]: the created Selector instance
        """
        raise NotImplementedError