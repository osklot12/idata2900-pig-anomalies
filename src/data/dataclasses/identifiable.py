from abc import ABC, abstractmethod

class Identifiable(ABC):
    """An entity with an ID."""

    @abstractmethod
    def get_id(self) -> str:
        """
        Returns the ID.

        Returns:
            str: the ID
        """
        raise NotImplementedError