from abc import ABC, abstractmethod

from src.data.dataset.entities.video_file import VideoFile


class VideoFileProvider(ABC):
    """An interface for video file providers."""

    @abstractmethod
    def get_video_file(self, video_id: str) -> VideoFile:
        """
        Returns a video file for the given video id.

        Args:
            video_id (str): the video id

        Returns:
            VideoFile: the video file
        """
        raise NotImplementedError