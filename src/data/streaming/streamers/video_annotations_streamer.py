import queue
from typing import Optional

from src.data.dataclasses.frame_annotations import FrameAnnotations
from src.data.dataset.entities.video_annotations import VideoAnnotations
from src.data.preprocessing.normalization.normalizers.bbox_normalizer import BBoxNormalizer
from src.data.pipeline.consumer import Consumer
from src.data.streaming.streamers.annotation_streamer import AnnotationStreamer


class VideoAnnotationsStreamer(AnnotationStreamer):
    """A streamer for streaming video annotations data."""

    def __init__(self, annotations: VideoAnnotations, consumer: Consumer[FrameAnnotations],
                 bbox_normalizer: BBoxNormalizer = None):
        """
        Initializes an VideoAnnotationStreamer instance.

        Args:
            annotations (VideoAnnotations): the video annotation data
            consumer (Consumer[Frame]): the consumer of the streaming data
            bbox_normalizer (BBoxNormalizer): the bounding box normalization strategy
        """
        super().__init__(consumer, bbox_normalizer)
        self._annotations = annotations
        self._frame_queue = queue.Queue()

    def _setup_stream(self) -> None:
        frame_annotations = self._annotations.get_data()

        for frame_annotation in frame_annotations:
            self._frame_queue.put(frame_annotation)

    def _get_next_annotation(self) -> Optional[FrameAnnotations]:
        annotation = None

        if not self._frame_queue.empty():
            annotation = self._frame_queue.get()

        return annotation