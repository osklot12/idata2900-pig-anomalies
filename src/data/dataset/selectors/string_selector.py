from abc import ABC, abstractmethod
from typing import List, Optional


class StringSelector(ABC):
    """Strategy for selecting strings."""

    @abstractmethod
    def next(self) -> Optional[str]:
        """
        Selects a string.

        Returns:
            Optional[str]: the selected file, or None if no strings are available
        """
        raise NotImplementedError