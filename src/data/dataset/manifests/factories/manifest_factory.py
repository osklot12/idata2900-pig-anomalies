from abc import ABC, abstractmethod

from src.data.dataset.manifests.manifest import Manifest


class ManifestFactory(ABC):
    """Interface for Manifest factories."""

    @abstractmethod
    def create_manifest(self) -> Manifest:
        """
        Creates and returns a Manifest instance.

        Returns:
            Manifest: the new Manifest instance
        """
        raise NotImplementedError