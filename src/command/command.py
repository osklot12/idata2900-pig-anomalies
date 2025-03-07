from abc import ABC, abstractmethod

class Command(ABC):
    """An interface for the command pattern."""

    @abstractmethod
    def execute(self):
        """Executes the command."""
        raise NotImplementedError