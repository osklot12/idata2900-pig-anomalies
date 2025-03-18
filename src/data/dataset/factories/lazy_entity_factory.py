from src.data.dataset.entities.lazy_video_annotations import LazyVideoAnnotations
from src.data.dataset.entities.lazy_video_file import LazyVideoFile
from src.data.dataset.entities.video_annotations import VideoAnnotations
from src.data.dataset.entities.video_file import VideoFile
from src.data.dataset.factories.dataset_entity_factory import DatasetEntityFactory
from src.data.loading.factories.loader_factory import LoaderFactory


class LazyEntityFactory(DatasetEntityFactory):
    """A factory for creating lazy dataset entities."""

    def __init__(self, loader_factory: LoaderFactory):
        """
        Initializes a LazyEntityFactory instance.

        Args:
            loader_factory (LoaderFactory): the loader factory
        """
        self._loader_factory = loader_factory

    def create_video_file(self, video_id: str) -> VideoFile:
        return LazyVideoFile(video_id, self._loader_factory.create_video_loader())

    def create_video_annotations(self, annotations_id: str) -> VideoAnnotations:
        return LazyVideoAnnotations(annotations_id, self._loader_factory.create_annotation_loader())