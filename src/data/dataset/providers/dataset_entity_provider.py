from abc import ABC, abstractmethod

from src.data.dataset.entities.video_annotations import VideoAnnotations
from src.data.dataset.entities.video_file import VideoFile


class DatasetEntityProvider(ABC):
    """Interface for dataset entity providers."""

    @abstractmethod
    def get_video_file(self, source: str) -> VideoFile:
        """
        Creates a VideoFile instance.

        Args:
            source (str): the source of the video file

        Returns:
            VideoFile: the VideoFile instance
        """
        raise NotImplementedError

    @abstractmethod
    def get_annotations_file(self, source: str) -> VideoAnnotations:
        """
        Creates a VideoAnnotations instance.

        Args:
            source (str): the source of the video file

        Returns:
            VideoAnnotations: the VideoAnnotations instance
        """
        raise NotImplementedError