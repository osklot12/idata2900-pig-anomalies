from abc import ABC, abstractmethod

from src.data.dataset.entities.video_file import VideoFile


class VideoFileLoader(ABC):
    """An interface for video file loaders."""

    @abstractmethod
    def load_video_file(self, video_id: str) -> VideoFile:
        """
        Loads a video file.

        Args:
            video_id (str): the video file identifier

        Returns:
            VideoFile: the video file
        """
        raise NotImplementedError