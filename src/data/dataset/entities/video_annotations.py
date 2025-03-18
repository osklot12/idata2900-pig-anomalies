from abc import ABC, abstractmethod
from typing import List

from src.data.dataclasses.frame_annotation import FrameAnnotation


class VideoAnnotations(ABC):
    """An interface for video annotations."""

    @abstractmethod
    def get_id(self) -> str:
        """
        Returns the annotation ID.

        Returns:
            str: the annotation ID
        """
        raise NotImplementedError

    @abstractmethod
    def get_data(self) -> List[FrameAnnotation]:
        """
        Returns the annotation data.

        Returns:
            List[FrameAnnotation]: list of annotations for video frames
        """
        raise NotImplementedError