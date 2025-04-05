from abc import ABC, abstractmethod

from src.data.dataset.registries.file_registry import FileRegistry
from src.data.loading.loaders.video_annotations_loader import VideoAnnotationsLoader
from src.data.loading.loaders.video_file_loader import VideoFileLoader


class LoaderFactory(ABC):
    """An abstract factory for creating loaders."""

    @abstractmethod
    def create_video_loader(self) -> VideoFileLoader:
        """
        Creates a video loader instance.

        Returns:
            VideoFileLoader: the video loader instance
        """
        raise NotImplementedError

    @abstractmethod
    def create_annotation_loader(self) -> VideoAnnotationsLoader:
        """
        Creates an annotation loader instance.

        Returns:
            VideoAnnotationsLoader: the annotation loader instance
        """
        raise NotImplementedError

    @abstractmethod
    def create_dataset_source(self) -> FileRegistry:
        """
        Creates a dataset source instance.

        Returns:
            FileRegistry: the dataset source instance
        """
        raise NotImplementedError