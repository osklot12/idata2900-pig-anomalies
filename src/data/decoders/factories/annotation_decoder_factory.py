from abc import ABC, abstractmethod

from src.data.decoders.annotation_decoder import AnnotationDecoder


class AnnotationDecoderFactory(ABC):
    """An interface for annotation decoder factories."""

    @abstractmethod
    def create_decoder(self) -> AnnotationDecoder:
        """
        Creates an AnnotationDecoder instance.

        Returns:
            AnnotationDecoder: the annotation decoder instance
        """
        raise NotImplementedError