from abc import ABC, abstractmethod

class Closable(ABC):
    """Interface for classes that can be closed."""

    @abstractmethod
    def close(self) -> None:
        """Closes the object."""
        raise NotImplementedError