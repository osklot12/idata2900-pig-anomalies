from typing import Callable, Optional, TypeVar

from src.data.dataclasses.frame import Frame
from src.data.dataclasses.frame_annotations import FrameAnnotations
from src.data.dataclasses.streamed_annotated_frame import StreamedAnnotatedFrame
from src.data.streaming.aggregators.buffered_aggregator import BufferedAggregator
from src.data.streaming.factories.streamer_pair_factory import StreamerPairFactory
from src.data.streaming.feedables.feedable import Feedable
from src.data.streaming.feedables.feedable_func import FeedableFunc
from src.data.streaming.streamers.ensemble_streamer import EnsembleStreamer
from src.data.streaming.streamers.streamer import Streamer
from src.data.streaming.streamers.streamer_status import StreamerStatus

T = TypeVar("T")

class AggregatedStreamer(Streamer):
    """A streamer consisting of a video and annotation streamer, aggregating the streamed data."""

    def __init__(self, streamer_factory: StreamerPairFactory, consumer: Feedable[StreamedAnnotatedFrame]):
        """
        Initializes an AggregatedStreamer instance.

        Args:
            streamer_factory (StreamerPairFactory): the factory used to create the streamers
            consumer (Feedable[StreamedAnnotatedFrame]): the consumer of the streaming data
        """
        self._aggregator = BufferedAggregator(consumer)
        streamer_pair = streamer_factory.create_streamer_pair(
            FeedableFunc[Frame](self._aggregator.feed_frame),
            FeedableFunc[FrameAnnotations](self._aggregator.feed_annotations)
        )

        if streamer_pair is None:
            raise RuntimeError("Failed to create streamers")

        self._ensemble_streamer = EnsembleStreamer(*streamer_pair)

    def start_streaming(self) -> None:
        print(f"[AggregatedStreamer] Started streaming...")
        self._ensemble_streamer.start_streaming()

    def stop_streaming(self) -> None:
        self._ensemble_streamer.stop_streaming()

    def wait_for_completion(self) -> None:
        self._ensemble_streamer.wait_for_completion()

    def get_status(self) -> StreamerStatus:
        return self._ensemble_streamer.get_status()
