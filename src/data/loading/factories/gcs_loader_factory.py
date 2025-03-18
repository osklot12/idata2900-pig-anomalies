from src.auth.factories.auth_service_factory import AuthServiceFactory
from src.data.dataset.sources.dataset_source import DatasetSource
from src.data.dataset.sources.gcs_dataset_source import GCSDatasetSource
from src.data.loading.factories.loader_factory import LoaderFactory
from src.data.loading.loaders.video_annotation_loader import VideoAnnotationsLoader
from src.data.loading.loaders.gcs_annotation_loader import GCSAnnotationLoader
from src.data.loading.loaders.gcs_video_loader import GCSVideoLoader
from src.data.loading.loaders.video_file_loader import VideoFileLoader


class GCSLoaderFactory(LoaderFactory):
    """A concrete factory for creating Google Cloud Storage loaders."""

    def __init__(self, bucket_name: str, auth_factory: AuthServiceFactory):
        """
        Initializes a GCSLoaderFactory instance.

        Args:
            bucket_name (str): the name of the gcs bucket
            auth_factory (AuthServiceFactory): the authentication service factory
        """
        self._bucket_name = bucket_name
        self._auth_factory = auth_factory

    def create_video_loader(self) -> VideoFileLoader:
        return GCSVideoLoader(self._bucket_name, self._auth_factory.create_auth_service())

    def create_annotation_loader(self) -> VideoAnnotationsLoader:
        return GCSAnnotationLoader(self._bucket_name, self._auth_factory.create_auth_service())

    def create_dataset_source(self) -> DatasetSource:
        return GCSDatasetSource(self._bucket_name, self._auth_factory.create_auth_service())