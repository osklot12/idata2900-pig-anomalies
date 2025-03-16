from abc import ABC, abstractmethod

class VideoLoader(ABC):
    """An interface for video loaders."""

    @abstractmethod
    def load_video(self, video_id: str) -> bytes:
        """
        Loads a video file.

        Args:
            video_id (str): the video identifier

        Returns:
            bytes: the video file in bytes
        """
        raise NotImplementedError