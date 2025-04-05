from typing import Callable, Optional

from src.data.dataclasses.streamed_annotated_frame import StreamedAnnotatedFrame
from src.data.loading.feed_status import FeedStatus
from src.data.parsing.factories.string_parser_factory import StringParserFactory
from src.data.streaming.factories.streamer_factory import StreamerFactory, T
from src.data.streaming.factories.streamer_pair_factory import StreamerPairFactory
from src.data.streaming.feedables.feedable import Feedable
from src.data.streaming.streamers.aggregated_streamer import AggregatedStreamer
from src.data.streaming.streamers.streamer import Streamer

class AggregatedStreamerFactory(StreamerFactory):
    """A factory for creating AggregatedStreamer instances."""

    def __init__(self, streamer_pair_factory: StreamerPairFactory):
        """
        Initializes an AggregatedStreamer instance.

        Args:
            streamer_pair_factory (StreamerPairFactory): a factory for creating streamer pairs
        """
        self._streamer_pair_factory = streamer_pair_factory

    def create_streamer(self, consumer: Feedable[StreamedAnnotatedFrame]) -> Optional[Streamer]:
        return AggregatedStreamer(self._streamer_pair_factory, consumer)