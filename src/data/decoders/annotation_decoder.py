from abc import ABC, abstractmethod
from typing import List

from src.data.dataclasses.frame_annotations import FrameAnnotations


class AnnotationDecoder(ABC):
    """An interface for annotation decoders."""

    @abstractmethod
    def decode(self, json_data: dict) -> List[FrameAnnotations]:
        """
        Decodes and returns the annotations.

        Args:
            json_data (dict): the raw annotations data

        Returns:
            List[FrameAnnotations]: the decoded annotations.
        """
        raise NotImplementedError