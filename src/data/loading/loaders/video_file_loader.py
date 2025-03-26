from abc import ABC, abstractmethod

class VideoFileLoader(ABC):
    """An interface for video file loaders."""

    @abstractmethod
    def load_video_file(self, video_id: str) -> bytes:
        """
        Loads video file data.

        Args:
            video_id (str): the ID of video to load

        Returns:
            bytes: the video file data
        """
        raise NotImplementedError