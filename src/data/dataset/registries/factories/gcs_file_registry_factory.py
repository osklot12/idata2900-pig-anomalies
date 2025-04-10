from src.auth.factories.auth_service_factory import AuthServiceFactory
from src.data.dataset.registries.factories.file_registry_factory import FileRegistryFactory
from src.data.dataset.registries.file_registry import FileRegistry
from src.data.dataset.registries.gcs_file_registry import GCSFileRegistry


class GCSFileRegistryFactory(FileRegistryFactory):
    """Factory for creating GCSFileRegistry instances."""

    def __init__(self, bucket_name: str, auth_factory: AuthServiceFactory):
        """
        Initializes a GCSFileRegistryFactory instance.

        Args:
            bucket_name (str): the name of the GCS bucket to use
            auth_factory (AuthServiceFactory): the auth service factory to use
        """
        self._bucket_name = bucket_name
        self._auth_factory = auth_factory

    def create_registry(self) -> FileRegistry:
        return GCSFileRegistry(self._bucket_name, self._auth_factory.create_auth_service())