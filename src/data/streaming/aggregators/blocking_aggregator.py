import threading
from typing import Optional

from src.data.dataclasses.frame import Frame
from src.data.dataclasses.frame_annotations import FrameAnnotations
from src.data.dataclasses.streamed_annotated_frame import StreamedAnnotatedFrame
from src.data.pipeline.consumer import Consumer
from src.data.pipeline.producer import Producer
from src.data.structures.atomic_bool import AtomicBool
from src.data.structures.atomic_var import AtomicVar

WAITING_TIMEOUT = 0.1

class BlockingAggregator(Producer[StreamedAnnotatedFrame]):
    """Aggregator of streaming data that blocks until a match is received."""

    def __init__(self, consumer: Optional[Consumer[StreamedAnnotatedFrame]] = None):
        """
        Initializes a BlockingAggregator instance.

        Args:
            consumer (Optional[Consumer[StreamedAnnotatedFrame]]): optional consumer of the aggregated data
        """
        self._frame: Optional[Frame] = None
        self._annotations: Optional[FrameAnnotations] = None

        self._consumer = AtomicVar[Consumer[StreamedAnnotatedFrame]](consumer)

        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)

    def feed_frame(self, frame: Optional[Frame], release: Optional[AtomicBool] = None) -> None:
        with self._condition:
            self._frame = frame
            self._condition.notify_all()

            match_found = False
            while not self._is_released(release) and not match_found:
                if self._annotations and self._annotations.index == frame.index:
                    self._consume()
                    match_found = True
                self._condition.wait(timeout=WAITING_TIMEOUT)

    def feed_annotations(self, annotations: Optional[FrameAnnotations], release: Optional[AtomicBool] = None) -> None:
        with self._condition:
            self._annotations = annotations
            self._condition.notify_all()

            match_found = False
            while not self._is_released(release) and not match_found:
                if self._frame and self._frame.index == annotations.index:
                    self._consume()
                    match_found = True
                self._condition.wait(timeout=WAITING_TIMEOUT)

    def _consume(self) -> None:
        """Feeds the current pair of frame and annotations to the consumer."""
        if self._frame and self._annotations and self._consumer.get():
            instance = self._create_instance(self._frame, self._annotations)
            self._frame = None
            self._annotations = None
            self._consumer.get().consume(instance)

    @staticmethod
    def _is_released(release: Optional[AtomicBool]) -> bool:
        """Checks whether the release is released."""
        return release is not None and release

    @staticmethod
    def _create_instance(frame: Frame, annotations: FrameAnnotations) -> StreamedAnnotatedFrame:
        """Creates a StreamedAnnotatedFrame instance."""
        return StreamedAnnotatedFrame(
            source=frame.source,
            index=frame.index,
            frame=frame.data,
            annotations=annotations.annotations,
        )

    def connect(self, consumer: Consumer[StreamedAnnotatedFrame]) -> None:
        self._consumer.set(consumer)