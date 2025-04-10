from typing import Optional

from src.data.dataclasses.streamed_annotated_frame import StreamedAnnotatedFrame
from src.data.streaming.aggregators.buffered_aggregator import BufferedAggregator
from src.data.streaming.factories.streamer_factory import StreamerFactory
from src.data.streaming.factories.streamer_pair_factory import StreamerPairFactory
from src.data.streaming.feedables.feedable import Feedable
from src.data.streaming.feedables.feedable_func import FeedableFunc
from src.data.streaming.streamers.ensemble_streamer import EnsembleStreamer
from src.data.streaming.streamers.streamer import Streamer


class AggregatedStreamerFactory(StreamerFactory[StreamedAnnotatedFrame]):
    """Streamer factory that creates pairs of video - annotations streamers and aggregates their output."""

    def __init__(self, streamer_pair_factory: StreamerPairFactory):
        """
        Initializes an AggregatedStreamerFactory instance.

        Args:
            streamer_pair_factory (StreamerPairFactory): factory for creating streamer pairs
        """
        self._pair_factory = streamer_pair_factory

    def create_streamer(self, consumer: Feedable[StreamedAnnotatedFrame]) -> Optional[Streamer]:
        streamer = None

        aggregator = BufferedAggregator(consumer)
        streamers = self._pair_factory.create_streamer_pair(
            frame_consumer=FeedableFunc(aggregator.feed_frame),
            annotations_consumer=FeedableFunc(aggregator.feed_annotations)
        )
        if streamers:
            video_streamer, annotations_streamer = streamers
            streamer = EnsembleStreamer(video_streamer, annotations_streamer)

        if streamer is None:
            print(f"[AggregatedStreamerFactory] End of stream")
        return streamer