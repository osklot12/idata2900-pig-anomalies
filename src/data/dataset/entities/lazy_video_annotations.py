from typing import List

from src.data.dataclasses.frame_annotations import FrameAnnotations
from src.data.dataset.entities.video_annotations import VideoAnnotations
from src.data.loading.loaders.video_annotations_loader import VideoAnnotationsLoader


class LazyVideoAnnotations(VideoAnnotations):
    """Video annotations that lazy loads the annotation data."""

    def __init__(self, annotation_id: str, annotation_loader: VideoAnnotationsLoader):
        """
        Initializes a LazyAnnotation instance.

        Args:
            annotation_id (str): the annotation ID
            annotation_loader (VideoAnnotationsLoader): the annotation loader
        """
        self._id = annotation_id
        self._loader = annotation_loader

    def get_id(self) -> str:
        return self._id

    def get_data(self) -> List[FrameAnnotations]:
        return self._loader.load_video_annotations(self._id)