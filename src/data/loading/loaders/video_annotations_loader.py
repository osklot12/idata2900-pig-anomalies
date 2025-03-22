from abc import ABC, abstractmethod
from typing import List

from src.data.dataclasses.frame_annotations import FrameAnnotations


class VideoAnnotationsLoader(ABC):
    """An interface for video annotations loaders."""

    @abstractmethod
    def load_video_annotations(self, annotations_id: str) -> List[FrameAnnotations]:
        """
        Loads video annotations data.

        Args:
            annotations_id (str): the ID of annotations to load

        Returns:
            List[FrameAnnotations]: the loaded video annotations
        """
        raise NotImplementedError