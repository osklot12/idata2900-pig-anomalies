from src.data.dataset.entities.lazy_video_annotations import LazyVideoAnnotations
from src.data.dataset.entities.lazy_video_file import LazyVideoFile
from src.data.dataset.entities.video_annotations import VideoAnnotations
from src.data.dataset.entities.video_file import VideoFile
from src.data.dataset.factories.dataset_entity_factory import DatasetEntityFactory
from src.data.loading.factories.loader_factory import LoaderFactory
from src.data.parsing.string_parser import StringParser


class LazyEntityFactory(DatasetEntityFactory):
    """A factory for creating lazy dataset entities."""

    def __init__(self, loader_factory: LoaderFactory, id_parser: StringParser):
        """
        Initializes a LazyEntityFactory instance.

        Args:
            loader_factory (LoaderFactory): the loader factory
            id_parser (StringParser): the parser for parsing instance IDs
        """
        self._loader_factory = loader_factory
        self._id_parser = id_parser

    def create_video_file(self, source: str) -> VideoFile:
        return LazyVideoFile(
            file_path=source,
            instance_id=self._id_parser.parse_string(source),
            video_loader=self._loader_factory.create_video_loader()
        )

    def create_video_annotations(self, source: str) -> VideoAnnotations:
        return LazyVideoAnnotations(
            file_path=source,
            instance_id=self._id_parser.parse_string(source),
            annotation_loader=self._loader_factory.create_annotation_loader()
        )