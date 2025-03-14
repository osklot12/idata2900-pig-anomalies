from abc import ABC, abstractmethod
from typing import Tuple, Optional

from src.data.dataclasses.dataset_file_pair import DatasetFilePair


class DatasetEntryProvider(ABC):
    """An interface for providers of paths of video-annotation file path pairs in a dataset."""

    @abstractmethod
    def get_random(self) -> Optional[DatasetFilePair]:
        """
        Returns video-annotation file path pair for a random dataset entry.

        Returns:
            Optional[DatasetFilePair]: tuple of video path and annotation path.
        """
        raise NotImplementedError