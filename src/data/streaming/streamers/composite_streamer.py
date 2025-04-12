from typing import Generic, TypeVar

from src.data.pipeline.consumer import Consumer
from src.data.pipeline.producer import Producer
from src.data.streaming.streamers.linear_streamer import LinearStreamer
from src.data.streaming.streamers.streamer_status import StreamerStatus

T = TypeVar("T")

class CompositeStreamer(Generic[T], LinearStreamer[T]):
    """Composition of streaming components, providing an abstract interface for connecting it in a pipeline."""

    def __init__(self, streamer: LinearStreamer, output: Producer[T]):
        """
        Initializes a CompositeStreamer instance.

        Args:
            streamer (LinearStreamer): the underlying streamer
            output (Producer[T]): the component outputting the streamed data
        """
        self._streamer = streamer
        self._output = output

    def start_streaming(self) -> None:
        self._streamer.start_streaming()

    def stop_streaming(self) -> None:
        self._streamer.stop_streaming()

    def wait_for_completion(self) -> None:
        self._streamer.wait_for_completion()

    def get_status(self) -> StreamerStatus:
        return self._streamer.get_status()

    def connect(self, consumer: Consumer[T]) -> None:
        self._output.connect(consumer)