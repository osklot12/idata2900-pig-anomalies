from typing import TypeVar, Optional

from src.data.dataclasses.streamed_annotated_frame import StreamedAnnotatedFrame
from src.data.dataset.providers.entity_factory import EntityFactory
from src.data.dataset.providers.instance_provider import InstanceProvider
from src.data.streaming.aggregators.buffered_aggregator import BufferedAggregator
from src.data.pipeline.consuming_func import ConsumingFunc
from src.data.streaming.streamers.composite_streamer import CompositeStreamer
from src.data.streaming.streamers.ensemble_streamer import EnsembleStreamer
from src.data.streaming.streamers.factories.streamer_factory import StreamerFactory
from src.data.streaming.streamers.linear_streamer import LinearStreamer
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

    def create_streamer(self) -> Optional[LinearStreamer]:
        streamer = None

        aggregator = BufferedAggregator()
        instance = self._instance_provider.get()
        if instance:
            video = self._entity_factory.create_video(instance.video_file)
            annotations = self._entity_factory.create_video_annotations(instance.annotation_file)
            if video and annotations:
                video_streamer = VideoFileStreamer(video, ConsumingFunc(aggregator.feed_frame))
                annotation_streamer = VideoAnnotationsStreamer(annotations, ConsumingFunc(aggregator.feed_annotations))

                streamer = CompositeStreamer(
                    streamer=EnsembleStreamer(video_streamer, annotation_streamer),
                    output=aggregator
                )

        return streamer