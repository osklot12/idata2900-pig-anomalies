from typing import Callable, Tuple, Optional

from src.data.dataclasses.frame import Frame
from src.data.dataclasses.frame_annotations import FrameAnnotations
from src.data.dataset.factories.dataset_entity_factory import DatasetEntityFactory
from src.data.dataset.providers.instance_provider import InstanceProvider
from src.data.loading.feed_status import FeedStatus
from src.data.preprocessing.normalization.factories.bbox_normalizer_factory import BBoxNormalizerFactory
from src.data.preprocessing.resizing.factories.frame_resizer_factory import FrameResizerFactory
from src.data.preprocessing.resizing.resizers.frame_resize_strategy import FrameResizeStrategy
from src.data.preprocessing.normalization.normalizers.bbox_normalizer import BBoxNormalizer
from src.data.streaming.factories.streamer_pair_factory import StreamerPairFactory
from src.data.streaming.feedables.feedable import Feedable
from src.data.streaming.streamers.annotation_streamer import AnnotationStreamer
from src.data.streaming.streamers.video_annotations_streamer import VideoAnnotationsStreamer
from src.data.streaming.streamers.video_file_streamer import VideoFileStreamer
from src.data.streaming.streamers.video_streamer import VideoStreamer


class FileStreamerPairFactory(StreamerPairFactory):
    """A factory for creating pairs of video file and annotation file streamers."""

    def __init__(self, instance_provider: InstanceProvider, entity_factory: DatasetEntityFactory,
                 frame_resizer_factory: FrameResizerFactory, bbox_normalizer_factory: BBoxNormalizerFactory):
        """
        Initializes a FileStreamerPairFactory instance.

        Args:
            instance_provider (InstanceProvider): a provider of dataset instances
            entity_factory (DatasetEntityFactory): a factory for creating the dataset entities
            frame_resizer_factory (FrameResizeStrategy): a factory for frame resizing strategies
            bbox_normalizer_factory (BBoxNormalizer): a factory for BBoxNormalization strategies
        """
        self._instance_provider = instance_provider
        self._entity_factory = entity_factory
        self._frame_resizer_factory = frame_resizer_factory
        self._bbox_normalizer_factory = bbox_normalizer_factory

    def create_streamer_pair(self, frame_consumer: Feedable[Frame], annotations_consumer: Feedable[FrameAnnotations]
                             ) -> Optional[Tuple[
        VideoStreamer, AnnotationStreamer]]:
        streamer_pair = None

        instance = self._instance_provider.next()
        if instance is not None:
            video = self._entity_factory.create_video_file(instance.video_file)
            annotations = self._entity_factory.create_video_annotations(instance.annotation_file)
            if video and annotations:
                video_streamer = VideoFileStreamer(video, frame_consumer, self._frame_resizer_factory.create_frame_resizer())
                annotation_streamer = VideoAnnotationsStreamer(annotations, annotations_consumer,
                                                               self._bbox_normalizer_factory.create_bbox_normalizer())
                streamer_pair = (video_streamer, annotation_streamer)

        return streamer_pair