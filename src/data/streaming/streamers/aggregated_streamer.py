from typing import Callable

from src.data.dataclasses.instance import Instance
from src.data.loading.feed_status import FeedStatus
from src.data.streaming.aggregators.buffered_instance_aggregator import BufferedInstanceAggregator
from src.data.streaming.streamers.annotation_streamer import AnnotationStreamer
from src.data.streaming.streamers.streamer import Streamer
from src.data.streaming.streamers.video_streamer import VideoStreamer


class AggregatedStreamer(Streamer):
    """A streamer consisting of a video and annotation streamer, aggregating the stream data."""

    def __init__(self, video_streamer: VideoStreamer, annotation_streamer: AnnotationStreamer,
                 callback: Callable[[Instance], FeedStatus]):
        """
        Initializes an AggregatedStreamer instance.

        Args:
            video_streamer (VideoStreamer): the video streamer
            annotation_streamer (AnnotationStreamer): the annotation streamer
            callback (Callable[[Instance], FeedStatus]): the callback function that will be fed with aggregated data
        """

    def start_streaming(self) -> None:
        pass

    def stop_streaming(self) -> None:
        pass

    def wait_for_completion(self) -> None:
        pass

