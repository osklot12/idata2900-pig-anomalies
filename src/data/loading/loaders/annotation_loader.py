from abc import ABC, abstractmethod

from src.data.dataset.entities.video_annotations import VideoAnnotations


class VideoAnnotationsLoader(ABC):
    """An interface for video annotations loaders."""

    @abstractmethod
    def load_video_annotations(self, annotation_id: str) -> VideoAnnotations:
        """
        Loads video annotations.

        Args:
            annotation_id (str): the video annotations identifier

        Returns:
            VideoAnnotations: the video annotations
        """
        raise NotImplementedError