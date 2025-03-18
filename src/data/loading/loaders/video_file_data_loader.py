from abc import ABC, abstractmethod

class VideoFileDataLoader(ABC):
    """An interface for video file data loaders."""

    @abstractmethod
    def load_video_file_data(self, video_id: str) -> bytes:
        """
        Loads video file data.

        Args:
            video_id (str): the ID of video to load

        Returns:
            bytes: the video file data
        """
        raise NotImplementedError