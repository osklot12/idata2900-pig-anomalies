from typing import Callable

from src.data.dataclasses.streamed_annotated_frame import StreamedAnnotatedFrame
from src.data.loading.feed_status import FeedStatus
from src.data.streaming.aggregators.buffered_instance_aggregator import BufferedInstanceAggregator
from src.data.streaming.factories.streamer_pair_factory import StreamerPairFactory
from src.data.streaming.streamers.ensemble_streamer import EnsembleStreamer
from src.data.streaming.streamers.streamer import Streamer


class AggregatedStreamer(Streamer):
    """A streamer consisting of a video and annotation streamer, aggregating the stream data."""

    def __init__(self, streamers_factory: StreamerPairFactory, callback: Callable[[StreamedAnnotatedFrame], FeedStatus]):
        """
        Initializes an AggregatedStreamer instance.

        Args:
            streamers_factory (StreamerPairFactory): the factory used to create the streamers
            callback (Callable[[Instance], FeedStatus]): the callback function that will be fed with aggregated data
        """
        self._aggregator = BufferedInstanceAggregator(callback)
        streamer_pair = streamers_factory.create_streamer_pair(
            self._aggregator.feed_frame,
            self._aggregator.feed_annotation
        )
        
        if streamer_pair is None:
            raise RuntimeError("Failed to create streamers")

        self._ensemble_streamer = EnsembleStreamer(*streamer_pair)

    def start_streaming(self) -> None:
        self._ensemble_streamer.start_streaming()

    def stop_streaming(self) -> None:
        self._ensemble_streamer.stop_streaming()

    def wait_for_completion(self) -> None:
        self._ensemble_streamer.wait_for_completion()

