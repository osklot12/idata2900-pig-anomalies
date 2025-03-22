from typing import Callable

from src.data.dataclasses.annotated_frame import AnnotatedFrame
from src.data.loading.feed_status import FeedStatus
from src.data.streaming.factories.streamer_factory import StreamerFactory
from src.data.streaming.factories.streamer_pair_factory import StreamerPairFactory
from src.data.streaming.streamers.aggregated_streamer import AggregatedStreamer


class AggregatedStreamerFactory(StreamerFactory):
    """A factory for creating AggregatedStreamer instances."""

    def __init__(self, streamer_pair_factory: StreamerPairFactory, callback: Callable[[AnnotatedFrame], FeedStatus]):
        """
        Initializes an AggregatedStreamer instance.

        Args:
            streamer_pair_factory (StreamerPairFactory): a factory for creating streamer pairs
            callback (Callable[[Instance], FeedStatus]): the callback to provide to the aggregated streamers
        """
        self._streamer_pair_factory = streamer_pair_factory
        self._callback = callback

    def create_streamer(self) -> AggregatedStreamer:
        return AggregatedStreamer(self._streamer_pair_factory, self._callback)