from abc import ABC, abstractmethod

from src.data.dataset.selectors.string_selector import StringSelector


class StringSelectorFactory(ABC):
    """Interface for StringSelector factories."""

    @abstractmethod
    def create_selector(self) -> StringSelector:
        """
        Creates and returns a new StrigSelector instance.

        Returns:
            StringSelector: the new StringSelector instance
        """
        raise NotImplementedError
