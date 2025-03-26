from abc import ABC, abstractmethod
from typing import List

from src.data.dataclasses.frame_annotations import FrameAnnotations
from src.data.dataset.entities.dataset_file import DatasetFile


class VideoAnnotations(DatasetFile, ABC):
    """An interface for video annotations."""

    @abstractmethod
    def get_data(self) -> List[FrameAnnotations]:
        """
        Returns the annotation data.

        Returns:
            List[FrameAnnotations]: list of annotations for video frames
        """
        raise NotImplementedError
