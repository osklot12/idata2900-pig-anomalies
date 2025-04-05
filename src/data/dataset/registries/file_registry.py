from abc import ABC, abstractmethod
from typing import List

class FileRegistry(ABC):
    """An interface for dataset file registries."""

    @abstractmethod
    def get_file_paths(self) -> set[str]:
        """
        Returns a set of file paths.

        Returns:
            set[str]: set of file paths.
        """
        raise NotImplementedError