from abc import ABC, abstractmethod

from src.data.dataset.entities.video_annotations import VideoAnnotations


class VideoAnnotationsProvider(ABC):
    """An interface for video annotations providers."""

    @abstractmethod
    def get_video_annotations(self, annotations_id: str) -> VideoAnnotations:
        """
        Returns video annotations for the given video annotations' id.

        Args:
            annotations_id (str): the video annotations id

        Returns:
            VideoAnnotations: the video annotations
        """
        raise NotImplementedError