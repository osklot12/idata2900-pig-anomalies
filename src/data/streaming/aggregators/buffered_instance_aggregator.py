from typing import Callable, Tuple, Optional, List
import numpy as np
from threading import Lock

from src.data.streaming.aggregators.instance_aggregator import InstanceAggregator
from src.data.data_structures.hash_buffer import HashBuffer
from src.data.dataclasses.annotation import Annotation
from src.data.dataclasses.frame import Frame
from src.data.dataclasses.instance import Instance
from src.data.loading.feed_status import FeedStatus
from src.typevars.enum_type import T_Enum


class BufferedInstanceAggregator(InstanceAggregator):
    """Buffers incoming frames and annotations and feeds forward an aggregated instance once matched."""

    def __init__(self, callback: Callable[[Instance], FeedStatus], buffer_size: int = 1000):
        """
        Initializes an instance of BufferedInstanceAggregator.

        Args:
            callback (Callable): Function that will be called and fed with a matched frame-annotation pair.
            buffer_size (int): The maximum capacity of the buffer.
        """
        self.callback = callback
        self.lock = Lock()

        # frame buffer
        self.frame_buffer = HashBuffer[
            Tuple[str, np.ndarray, bool]
        ](max_size=buffer_size)

        # annotation buffer
        self.annotation_buffer = HashBuffer[
            Tuple[str, Optional[List[Tuple[T_Enum, float, float, float, float]]], bool]
        ](max_size=buffer_size)

    def feed_frame(self, frame: Frame) -> FeedStatus:
        with self.lock:
            result = FeedStatus.DROP
            index = frame.index

            if frame.end_of_stream:
                index = -1

            if self._match_frame(frame):
                result = FeedStatus.ACCEPT

            else:
                # store frame until annotation arrives
                self.frame_buffer.add(
                    index,
                    (frame.source, frame.data, frame.end_of_stream)
                )
                result = FeedStatus.ACCEPT

        return result

    def feed_annotation(self, annotation: Annotation) -> FeedStatus:
        with self.lock:
            result = FeedStatus.DROP
            index = annotation.index

            if annotation.end_of_stream:
                index = -1

            if self._match_annotation(annotation):
                result = FeedStatus.ACCEPT

            else:
                # store annotation until frame arrives
                self.annotation_buffer.add(
                    index,
                    (annotation.source, annotation.annotations, annotation.end_of_stream)
                )
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
            annotations, annotation_end = self.annotation_buffer.pop(frame.index)[1:]
            instance = Instance(
                frame.source,
                frame.index,
                frame.data,
                annotations,
                frame.end_of_stream
            )
            self._feed_instance(instance)
            result=True

        return result


    def _match_annotation(self, annotation: Annotation) -> bool:
        """
        Tries to match an annotation with an existing frame.
        If successful, the paired instance will be fed forward.

        Args:
            annotation (Annotation): The annotation to match.

        Returns:
            True if successful, False otherwise.
        """
        result = False

        if self.frame_buffer.has(annotation.index):
            # match found
            frame_data, frame_end = self.frame_buffer.pop(annotation.index)[1:]
            instance = Instance(
                annotation.source,
                annotation.index,
                frame_data,
                annotation.annotations,
                annotation.end_of_stream
            )
            self._feed_instance(instance)
            result = True

        return result

    def _feed_instance(self, instance: Instance) -> FeedStatus:
        """
        Feeds forward an instance of a matched frame-annotation pair to the registered callback.

        Args:
            instance (Instance): The instance to feed forward.

        Returns:
            FeedStatus: The status of the feed forward.
        """
        print(f"[FrameAnnotationLoader] Matched frame and annotation pair with index {instance.index}, feeding forward")
        feed_result = FeedStatus.RETRY_LATER

        keep_feeding = True
        # keep feeding until fed or rejected
        while keep_feeding:
            feed_result = self.callback(instance)
            if feed_result != FeedStatus.RETRY_LATER:
                keep_feeding = False

        return feed_result