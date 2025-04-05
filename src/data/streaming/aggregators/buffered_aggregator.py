from typing import Callable, Tuple, Optional, List
import numpy as np
from threading import Lock

from src.data.parsing.string_parser import StringParser
from src.data.streaming.aggregators.aggregator import Aggregator
from src.data.streaming.feedables.feedable import Feedable
from src.data.structures.hash_buffer import HashBuffer
from src.data.dataclasses.frame_annotations import FrameAnnotations
from src.data.dataclasses.frame import Frame
from src.data.dataclasses.streamed_annotated_frame import StreamedAnnotatedFrame
from src.data.loading.feed_status import FeedStatus
from src.typevars.enum_type import T_Enum

END_OF_STREAM_INDEX = -1

class BufferedAggregator(Aggregator):
    """Buffers incoming frames and annotations and feeds forward an aggregated instance once matched."""

    def __init__(self, consumer: Feedable[StreamedAnnotatedFrame], buffer_size: int = 1000):
        """
        Initializes aBufferedInstanceAggregator instance.

        Args:
            consumer (Feedable[StreamedAnnotatedFrame]): the consumer of the aggregated data
            buffer_size (int): The maximum capacity of the buffer.
        """
        self._consumer = consumer
        self._lock = Lock()

        self._frame_buffer = HashBuffer[int, Optional[Frame]](max_size=buffer_size)
        self._annotation_buffer = HashBuffer[int, Optional[FrameAnnotations]](max_size=buffer_size)

    def feed_frame(self, frame: Frame) -> None:
        with self._lock:
            if frame is not None:
                if self._annotation_buffer.has(frame.index):
                    self._feed_consumer(frame, self._annotation_buffer.pop(frame.index))
                else:
                    self._frame_buffer.add(frame.index, frame)

            else:
                if self._annotation_buffer.has(END_OF_STREAM_INDEX):
                    self._consumer.feed(self._annotation_buffer.pop(END_OF_STREAM_INDEX))
                else:
                    self._frame_buffer.add(END_OF_STREAM_INDEX, None)


    def feed_annotations(self, annotations: FrameAnnotations) -> None:
        with self._lock:
            if annotations is not None:
                if self._frame_buffer.has(annotations.index):
                    self._feed_consumer(self._frame_buffer.pop(annotations.index), annotations)
                else:
                    self._annotation_buffer.add(annotations.index, annotations)

            else:
                if self._frame_buffer.has(END_OF_STREAM_INDEX):
                    self._consumer.feed(self._frame_buffer.pop(END_OF_STREAM_INDEX))
                else:
                    self._annotation_buffer.add(END_OF_STREAM_INDEX, None)

    def _feed_consumer(self, frame: Frame, anno: FrameAnnotations) -> None:
        """Feeds the consumer with a StreamedAnnotatedFrame instance."""
        self._consumer.feed(
            StreamedAnnotatedFrame(
                source=frame.source,
                index=frame.index,
                frame=frame.data,
                annotations=anno.annotations
            )
        )