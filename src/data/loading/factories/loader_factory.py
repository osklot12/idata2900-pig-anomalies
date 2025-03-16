from abc import ABC, abstractmethod

from src.data.dataset.dataset_source import DatasetSource
from src.data.loading.loaders.annotation_loader import AnnotationLoader
from src.data.loading.loaders.video_loader import VideoLoader


class LoaderFactory(ABC):
    """An abstract factory for creating loaders."""

    @abstractmethod
    def create_video_loader(self) -> VideoLoader:
        """
        Creates a video loader instance.

        Returns:
            VideoLoader: the video loader instance
        """
        raise NotImplementedError

    @abstractmethod
    def create_annotation_loader(self) -> AnnotationLoader:
        """
        Creates an annotation loader instance.

        Returns:
            AnnotationLoader: the annotation loader instance
        """
        raise NotImplementedError

    @abstractmethod
    def create_dataset_source(self) -> DatasetSource:
        """
        Creates a dataset source instance.

        Returns:
            DatasetSource: the dataset source instance
        """
        raise NotImplementedError