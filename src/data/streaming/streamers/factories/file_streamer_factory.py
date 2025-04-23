from typing import TypeVar, Optional

from src.data.dataclasses.annotated_frame import AnnotatedFrame
from src.data.dataset.providers.entity_factory import EntityFactory
from src.data.dataset.providers.instance_provider import InstanceProvider
from src.data.pipeline.consuming_rfunc import ConsumingRFunc
from src.data.streaming.aggregators.blocking_aggregator import BlockingAggregator
from src.data.streaming.aggregators.buffered_aggregator import BufferedAggregator
from src.data.pipeline.consuming_func import ConsumingFunc
from src.data.streaming.streamers.composite_streamer import CompositeStreamer
from src.data.streaming.streamers.ensemble_streamer import EnsembleStreamer
from src.data.streaming.streamers.factories.streamer_factory import StreamerFactory
from src.data.streaming.streamers.producer_streamer import ProducerStreamer
from src.data.streaming.streamers.video_annotations_streamer import VideoAnnotationsStreamer
from src.data.streaming.streamers.video_file_streamer import VideoFileStreamer


class FileStreamerFactory(StreamerFactory[AnnotatedFrame]):
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

    def create_streamer(self) -> Optional[ProducerStreamer[AnnotatedFrame]]:
        streamer = None

        aggregator = BlockingAggregator()
        instance = self._instance_provider.get()
        if instance:
            video = self._entity_factory.create_video(instance.video_file)
            annotations = self._entity_factory.create_video_annotations(instance.annotation_file)
            if video and annotations:
                video_streamer = VideoFileStreamer(video)
                video_consumer = ConsumingRFunc(aggregator.feed_frame, video_streamer.get_releaser())
                video_streamer.connect(video_consumer)

                annotations_streamer = VideoAnnotationsStreamer(annotations)
                annotations_consumer = ConsumingRFunc(aggregator.feed_annotations, annotations_streamer.get_releaser())
                annotations_streamer.connect(annotations_consumer)

                streamer = CompositeStreamer(
                    streamer=EnsembleStreamer(video_streamer, annotations_streamer),
                    output=aggregator
                )

        return streamer