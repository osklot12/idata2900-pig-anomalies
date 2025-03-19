from abc import ABC, abstractmethod
from typing import List

from src.data.dataclasses.frame_annotation import FrameAnnotation


class AnnotationDecoder(ABC):
    """An interface for annotation decoders."""

    @abstractmethod
    def decode_annotations(self) -> List[FrameAnnotation]:
        """
        Decodes and returns the annotations.

        Returns:
            List[FrameAnnotation]: the decoded annotations.
        """
        raise NotImplementedError