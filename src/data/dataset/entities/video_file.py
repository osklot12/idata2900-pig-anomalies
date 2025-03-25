from abc import ABC, abstractmethod

from src.data.dataset.entities.dataset_file import DatasetFile


class VideoFile(DatasetFile, ABC):
    """An interface for video files."""

    @abstractmethod
    def get_data(self) -> bytes:
        """
        Returns the video data.

        Returns:
            bytes: the video data
        """
        raise NotImplementedError
