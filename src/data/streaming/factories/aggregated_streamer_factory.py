from typing import Callable, Optional

from src.data.dataclasses.streamed_annotated_frame import StreamedAnnotatedFrame
from src.data.loading.feed_status import FeedStatus
from src.data.parsing.factories.string_parser_factory import StringParserFactory
from src.data.streaming.factories.streamer_factory import StreamerFactory
from src.data.streaming.factories.streamer_pair_factory import StreamerPairFactory
from src.data.streaming.streamers.aggregated_streamer import AggregatedStreamer


class AggregatedStreamerFactory(StreamerFactory):
    """A factory for creating AggregatedStreamer instances."""

    def __init__(self, streamer_pair_factory: StreamerPairFactory,
                 callback: Callable[[StreamedAnnotatedFrame], FeedStatus],
                 source_parser_factory: Optional[StringParserFactory] = None):
        """
        Initializes an AggregatedStreamer instance.

        Args:
            streamer_pair_factory (StreamerPairFactory): a factory for creating streamer pairs
            callback (Callable[[Instance], FeedStatus]): the callback to provide to the aggregated streamers
            source_parser_factory (Optional[StringParserFactory]): an optional factory for creating parsers for the source id
        """
        self._streamer_pair_factory = streamer_pair_factory
        self._callback = callback
        self._source_parser_factory = source_parser_factory

    def create_streamer(self) -> AggregatedStreamer:
        source_parser = None
        if self._source_parser_factory:
            source_parser = self._source_parser_factory.create_string_parser()

        if source_parser:
            product = AggregatedStreamer(self._streamer_pair_factory, self._callback, source_parser)
        else:
            product = AggregatedStreamer(self._streamer_pair_factory, self._callback)

        return product