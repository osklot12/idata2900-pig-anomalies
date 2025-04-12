from typing import Optional
from threading import Lock

from src.data.pipeline.producer import Producer, T
from src.data.streaming.aggregators.aggregator import Aggregator
from src.data.pipeline.consumer import Consumer
from src.data.structures.atomic_var import AtomicVar
from src.data.structures.hash_buffer import HashBuffer
from src.data.dataclasses.frame_annotations import FrameAnnotations
from src.data.dataclasses.frame import Frame
from src.data.dataclasses.streamed_annotated_frame import StreamedAnnotatedFrame

END_OF_STREAM_INDEX = -1

class BufferedAggregator(Aggregator, Producer[StreamedAnnotatedFrame]):
    """Buffers incoming frames and annotations and feeds forward an aggregated instance once matched."""

    def __init__(self, buffer_size: int = 1000, consumer: Optional[Consumer[StreamedAnnotatedFrame]] = None):
        """
        Initializes aBufferedInstanceAggregator instance.

        Args:
            buffer_size (int): The maximum capacity of the buffer.
            consumer (Optional[Consumer[StreamedAnnotatedFrame]]): optional consumer of the aggregated data
        """
        self._consumer = AtomicVar[Consumer[StreamedAnnotatedFrame]](consumer)
        self._lock = Lock()

        self._frame_buffer = HashBuffer[int, Optional[Frame]](max_size=buffer_size)
        self._annotation_buffer = HashBuffer[int, Optional[FrameAnnotations]](max_size=buffer_size)

    def feed_frame(self, frame: Optional[Frame]) -> None:
        with self._lock:
            if frame is not None:
                if self._annotation_buffer.has(frame.index):
                    self._feed_consumer(frame, self._annotation_buffer.pop(frame.index))
                else:
                    self._frame_buffer.add(frame.index, frame)

            else:
                if self._annotation_buffer.has(END_OF_STREAM_INDEX):
                    self._consumer.consume(self._annotation_buffer.pop(END_OF_STREAM_INDEX))
                else:
                    self._frame_buffer.add(END_OF_STREAM_INDEX, None)

    def feed_annotations(self, annotations: Optional[FrameAnnotations]) -> None:
        with self._lock:
            if annotations is not None:
                if self._frame_buffer.has(annotations.index):
                    self._feed_consumer(self._frame_buffer.pop(annotations.index), annotations)
                else:
                    self._annotation_buffer.add(annotations.index, annotations)

            else:
                if self._frame_buffer.has(END_OF_STREAM_INDEX):
                    self._consumer.consume(self._frame_buffer.pop(END_OF_STREAM_INDEX))
                else:
                    self._annotation_buffer.add(END_OF_STREAM_INDEX, None)


    def _feed_consumer(self, frame: Frame, anno: FrameAnnotations) -> None:
        """Feeds the consumer with a StreamedAnnotatedFrame instance."""
        self._consumer.consume(
            StreamedAnnotatedFrame(
                source=frame.source,
                index=frame.index,
                frame=frame.data,
                annotations=anno.annotations
            )
        )

    def connect(self, consumer: Consumer[StreamedAnnotatedFrame]) -> None:
        self.