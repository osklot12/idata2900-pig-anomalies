from abc import ABC, abstractmethod

from src.data.dataset.providers.dataset_entity_provider import DatasetEntityProvider


class EntityProviderFactory(ABC):
    """Interface for entity provider factories."""

    @abstractmethod
    def create_provider(self) -> DatasetEntityProvider:
        """
        Creates and returns a new DatasetEntityProvider instance.

        Returns:
            DatasetEntityProvider: the new DatasetEntityProvider instance
        """
        raise NotImplementedError
