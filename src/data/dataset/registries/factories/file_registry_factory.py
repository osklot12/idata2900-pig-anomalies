from abc import ABC, abstractmethod

from src.data.dataset.registries.file_registry import FileRegistry


class FileRegistryFactory(ABC):
    """Interface for Registry factories."""

    @abstractmethod
    def create_registry(self) -> FileRegistry:
        """
        Creates and returns a new FileRegistry instance.

        Returns:
            FileRegistry: the new FileRegistry instance
        """
        raise NotImplementedError