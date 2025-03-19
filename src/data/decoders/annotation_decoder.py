from abc import ABC, abstractmethod
from typing import List

from src.data.dataclasses.frame_annotation import FrameAnnotation


class AnnotationDecoder(ABC):
    """An interface for annotation decoders."""

    @abstractmethod
    def decode_annotations(self, raw_data: bytes) -> List[FrameAnnotation]:
        """
        Decodes and returns the annotations.

        Args:
            raw_data (bytes): the raw annotations data

        Returns:
            List[FrameAnnotation]: the decoded annotations.
        """
        raise NotImplementedError