from src.data.dataset.manifests.factories.manifest_factory import ManifestFactory
from src.data.dataset.manifests.manifest import Manifest
from src.data.dataset.manifests.matching_manifest import MatchingManifest
from src.data.dataset.registries.factories.file_registry_factory import FileRegistryFactory


class MatchingManifestFactory(ManifestFactory):
    """Factory for creating MatchingManifest instances."""

    def __init__(self, video_registry_factory: FileRegistryFactory, annotations_registry_factory: FileRegistryFactory):
        """
        Initializes a MatchingManifestFactory instance.

        Args:
            video_registry_factory (FileRegistryFactory): factory for creating video registries
            annotations_registry_factory (FileRegistryFactory): factory for creating annotations registries
        """
        self._video_registry_factory = video_registry_factory
        self._annotations_registry_factory = annotations_registry_factory

    def create_manifest(self) -> Manifest:
        return MatchingManifest(
            video_registry=self._video_registry_factory.create_registry(),
            annotations_registry=self._annotations_registry_factory.create_registry()
        )