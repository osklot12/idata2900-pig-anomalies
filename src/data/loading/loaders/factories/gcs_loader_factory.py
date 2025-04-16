from src.auth.factories.auth_service_factory import AuthServiceFactory
from src.data.dataset.registries.file_registry import FileRegistry
from src.data.dataset.registries.gcs_file_registry import GCSFileRegistry
from src.data.decoders.factories.annotation_decoder_factory import AnnotationDecoderFactory
from src.data.loading.loaders.factories.loader_factory import LoaderFactory
from src.data.loading.loaders.gcs_annotation_loader import GCSAnnotationLoader
from src.data.loading.loaders.gcs_video_loader import GCSVideoLoader
from src.data.loading.loaders.video_annotations_loader import VideoAnnotationsLoader
from src.data.loading.loaders.video_file_loader import VideoFileLoader


class GCSLoaderFactory(LoaderFactory):
    """A concrete factory for creating Google Cloud Storage loaders."""

    def __init__(self, bucket_name: str, auth_factory: AuthServiceFactory, decoder_factory: AnnotationDecoderFactory):
        """
        Initializes a GCSLoaderFactory instance.

        Args:
            bucket_name (str): the name of the gcs bucket
            auth_factory (AuthServiceFactory): the authentication service factory
            decoder_factory (AnnotationDecoderFactory): the annotation decoder factory
        """
        self._bucket_name = bucket_name
        self._auth_factory = auth_factory
        self._decoder_factory = decoder_factory

    def create_video_loader(self) -> VideoFileLoader:
        return GCSVideoLoader(self._bucket_name, self._auth_factory.create_auth_service())

    def create_annotation_loader(self) -> VideoAnnotationsLoader:
        return GCSAnnotationLoader(
            bucket_name=self._bucket_name,
            auth_service=self._auth_factory.create_auth_service(),
            decoder=self._decoder_factory.create_decoder()
        )

    def create_file_registry(self) -> FileRegistry:
        return GCSFileRegistry(self._bucket_name, self._auth_factory.create_auth_service())