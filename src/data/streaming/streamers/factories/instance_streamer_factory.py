from typing import TypeVar, Optional

from src.data.dataclasses.streamed_annotated_frame import StreamedAnnotatedFrame
from src.data.dataset.providers.entity_factory import EntityFactory
from src.data.dataset.providers.instance_provider import InstanceProvider
from src.data.streaming.aggregators.buffered_aggregator import BufferedAggregator
from src.data.pipeline.consumer import Consumer
from src.data.streaming.feedables.feedable_func import FeedableFunc
from src.data.streaming.streamers.ensemble_streamer import EnsembleStreamer
from src.data.streaming.streamers.factories.streamer_factory import StreamerFactory
from src.data.streaming.streamers.streamer import Streamer
from src.data.streaming.streamers.video_annotations_streamer import VideoAnnotationsStreamer
from src.data.streaming.streamers.video_file_streamer import VideoFileStreamer

T = TypeVar("T")


class InstanceStreamerFactory(StreamerFactory[StreamedAnnotatedFrame]):
    """Factory for creating dataset instance streamers."""

    def __init__(self, instance_provider: InstanceProvider, entity_factory: EntityFactory):
        """
        Initializes an InstanceStreamerFactory instance.

        Args:
            instance_provider (InstanceProvider): provider of dataset instances
            entity_factory (EntityFactory): factory for creating dataset entities
        """
        self._instance_provider = instance_provider
        self._entity_factory = entity_factory

    def create_streamer(self, consumer: Consumer[StreamedAnnotatedFrame]) -> Optional[Streamer]:
        streamer = None

        aggregator = BufferedAggregator(consumer)
        instance = self._instance_provider.get()
        if instance:
            video = self._entity_factory.create_video(instance.video_file)
            annotations = self._entity_factory.create_video_annotations(instance.annotation_file)
            if video and annotations:
                video_streamer = VideoFileStreamer(video, FeedableFunc(aggregator.feed_frame))
                annotation_streamer = VideoAnnotationsStreamer(annotations, FeedableFunc(aggregator.feed_annotations))

                streamer = EnsembleStreamer(video_streamer, annotation_streamer)

        return streamer