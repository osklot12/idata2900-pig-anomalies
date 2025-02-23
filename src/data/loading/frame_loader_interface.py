from abc import ABC, abstractmethod

from src.data.dataset_source import DatasetSource


class FrameLoaderInterface(ABC):
    """An interface for a frame loader."""

    @abstractmethod
    def load_frames(self, video_blob_name: str):
        """Loads frames of a given video."""
        pass

    @abstractmethod
    def get_data_source(self) -> "DatasetSource":
        """Returns the underlying data source."""
        pass