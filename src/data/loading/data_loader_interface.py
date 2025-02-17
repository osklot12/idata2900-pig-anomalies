from abc import ABC, abstractmethod
from io import BytesIO
from typing import List

class DataLoaderInterface(ABC):
    """
    Interface for a data loader, defining standard methods for fetching video data.
    """

    @abstractmethod
    def list_files(self, prefix: str = "", file_extension: str = "") -> List[str]:
        """Lists available files in the storage system."""
        pass

    @abstractmethod
    def get_video(self, blob_name: str) -> BytesIO:
        """Returns a video file as a BytesIO stream."""
        pass