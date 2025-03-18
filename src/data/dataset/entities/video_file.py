from abc import ABC, abstractmethod

class VideoFile(ABC):
    """An interface for video files."""

    @abstractmethod
    def get_id(self) -> str:
        """
        Returns the video ID.

        Returns:
            str: the video ID
        """
        raise NotImplementedError

    @abstractmethod
    def get_data(self) -> bytes:
        """
        Returns the video data.

        Returns:
            bytes: the video data
        """
        raise NotImplementedError