from typing import Callable, Tuple, Optional, List
import numpy as np
from threading import Lock

from src.data.parsing.string_parser import StringParser
from src.data.streaming.aggregators.instance_aggregator import InstanceAggregator
from src.data.data_structures.hash_buffer import HashBuffer
from src.data.dataclasses.frame_annotations import FrameAnnotations
from src.data.dataclasses.frame import Frame
from src.data.dataclasses.streamed_annotated_frame import StreamedAnnotatedFrame
from src.data.loading.feed_status import FeedStatus
from src.typevars.enum_type import T_Enum


class BufferedInstanceAggregator(InstanceAggregator):
    """Buffers incoming frames and annotations and feeds forward an aggregated instance once matched."""

    def __init__(self, callback: Callable[[StreamedAnnotatedFrame], FeedStatus], buffer_size: int = 1000):
        """
        Initializes an instance of BufferedInstanceAggregator.

        Args:
            callback (Callable): Function that will be called and fed with a matched frame-annotation pair.
            buffer_size (int): The maximum capacity of the buffer.
        """
        self.callback = callback
        self.lock = Lock()

        # frame buffer
        self.frame_buffer = HashBuffer[Frame](max_size=buffer_size)

        # annotation buffer
        self.annotation_buffer = HashBuffer[FrameAnnotations](max_size=buffer_size)

    def feed_frame(self, frame: Frame) -> FeedStatus:
        if frame is None:
            raise ValueError("frame cannot be None")

        with self.lock:
            result = FeedStatus.DROP
            index = frame.index

            if frame.end_of_stream:
                index = -1

            if self._match_frame(frame):
                result = FeedStatus.ACCEPT

            else:
                # store frame until annotation arrives
                self.frame_buffer.add(index, frame)
                result = FeedStatus.ACCEPT

        return result

    def feed_annotations(self, annotations: FrameAnnotations) -> FeedStatus:
        if annotations is None:
            raise ValueError("annotations cannot be None")

        with self.lock:
            result = FeedStatus.DROP
            index = annotations.index

            if annotations.end_of_stream:
                index = -1

            if self._match_annotation(annotations):
                result = FeedStatus.ACCEPT

            else:
                # store annotation until frame arrives
                self.annotation_buffer.add(index, annotations)
                result = FeedStatus.ACCEPT

        return result

    def _match_frame(self, frame: Frame) -> bool:
        """
        Tries to match a frame with an existing annotation.
        If successful, the paired instance will be fed forward.

        Args:
            frame (Frame): The frame to match.

        Returns:
            True if successful, False otherwise.
        """
        result = False

        if self.annotation_buffer.has(frame.index):
            # match found
            annotations = self.annotation_buffer.pop(frame.index)
            self._feed_forward(frame, annotations)
            result = True

        return result

    def _match_annotation(self, annotations: FrameAnnotations) -> bool:
        """
        Tries to match an annotation with an existing frame.
        If successful, the paired instance will be fed forward.

        Args:
            annotations (FrameAnnotations): The annotation to match.

        Returns:
            True if successful, False otherwise.
        """
        result = False

        if self.frame_buffer.has(annotations.index):
            # match found
            frame = self.frame_buffer.pop(annotations.index)
            self._feed_forward(frame, annotations)
            result = True

        return result

    def _feed_forward(self, frame: Frame, annotations: FrameAnnotations) -> None:
        """Feeds forward the matched frame and annotations."""
        source = frame.source

        instance = StreamedAnnotatedFrame(
            source=source,
            index=frame.index,
            frame=frame.data,
            annotations=annotations.annotations,
            end_of_stream=frame.end_of_stream
        )
        self._feed_instance(instance)

    def _feed_instance(self, instance: StreamedAnnotatedFrame) -> FeedStatus:
        """
        Feeds forward an instance of a matched frame-annotation pair to the registered callback.

        Args:
            instance (StreamedAnnotatedFrame): The instance to feed forward.

        Returns:
            FeedStatus: The status of the feed forward.
        """
        feed_result = FeedStatus.RETRY_LATER

        keep_feeding = True
        # keep feeding until fed or rejected
        while keep_feeding:
            feed_result = self.callback(instance)
            if feed_result != FeedStatus.RETRY_LATER:
                keep_feeding = False

        return feed_result
