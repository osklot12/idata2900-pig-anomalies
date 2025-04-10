from abc import ABC, abstractmethod

from src.data.dataset.providers.instance_provider import InstanceProvider


class InstanceProviderFactory(ABC):
    """Interface for instance provider factories."""

    @abstractmethod
    def create_provider(self) -> InstanceProvider:
        """
        Creates and returns an InstanceProvider instance.

        Returns:
            InstanceProvider: the new InstanceProvider instance
        """
        raise NotImplementedError