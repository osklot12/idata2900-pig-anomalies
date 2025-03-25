from abc import ABC, abstractmethod
from typing import List, Tuple

from src.data.dataclasses.frame_annotations import FrameAnnotations
from src.data.dataclasses.source_metadata import SourceMetadata


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
    def get_data(self) -> List[FrameAnnotations]:
        """
        Returns the annotation data.

        Returns:
            List[FrameAnnotations]: list of annotations for video frames
        """
        raise NotImplementedError