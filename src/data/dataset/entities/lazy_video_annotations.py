from typing import List

from src.data.dataclasses.frame_annotations import FrameAnnotations
from src.data.dataset.entities.dataset_file import DatasetFile
from src.data.dataset.entities.video_annotations import VideoAnnotations
from src.data.loading.loaders.video_annotations_loader import VideoAnnotationsLoader


class LazyVideoAnnotations(VideoAnnotations):
    """Video annotations that lazy loads the annotation data."""

    def __init__(self, file_path: str, instance_id: str, annotation_loader: VideoAnnotationsLoader):
        """
        Initializes a LazyAnnotation instance.

        Args:
            file_path (str): path to the annotation file
            instance_id (str): the instance ID
            annotation_loader (VideoAnnotationsLoader): the annotation loader
        """
        super().__init__(file_path, instance_id)
        self._loader = annotation_loader

    def get_data(self) -> List[FrameAnnotations]:
        return self._loader.load_video_annotations(self._instance_id)