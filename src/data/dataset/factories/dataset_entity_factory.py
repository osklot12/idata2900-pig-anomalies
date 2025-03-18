from abc import ABC, abstractmethod

from src.data.dataset.entities.video_annotations import VideoAnnotations
from src.data.dataset.entities.video_file import VideoFile


class DatasetEntityFactory(ABC):
    """Abstract factory for dataset entities."""

    @abstractmethod
    def create_video_file(self, video_id: str) -> VideoFile:
        """
        Creates a VideoFile instance.

        Args:
            video_id (str): the video ID

        Returns:
            VideoFile: the VideoFile instance
        """
        raise NotImplementedError

    @abstractmethod
    def create_video_annotations(self, annotations_id: str) -> VideoAnnotations:
        """
        Creates a VideoAnnotations instance.

        Args:
            annotations_id (str): the annotation ID

        Returns:
            VideoAnnotations: the VideoAnnotations instance
        """
        raise NotImplementedError