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
        self._eos = False

        self._consumer = AtomicVar[Consumer[StreamedAnnotatedFrame]](consumer)

        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)

    def feed_frame(self, frame: Optional[Frame], release: Optional[AtomicBool] = None) -> bool:
        match_found = False

        with self._condition:
            if frame is not None:
                self._frame = frame
                if self._annotations is None:
                    while self._frame is not None and not self._is_released(release):
                        self._condition.wait(timeout=WAITING_TIMEOUT)

                else:
                    self._consume()
                    match_found = True

            else:
                self._handle_eos()

        return match_found

    def feed_annotations(self, annotations: Optional[FrameAnnotations], release: Optional[AtomicBool] = None) -> bool:
        match_found = False

        with self._condition:
            if not annotations is None:
                self._annotations = annotations
                if self._frame is None:
                    while self._annotations is not None and not self._is_released(release):
                        self._condition.wait(timeout=WAITING_TIMEOUT)

                else:
                    self._consume()
                    match_found = True

            else:
                self._handle_eos()

        return match_found

    def _consume(self) -> None:
        """Feeds the current pair of frame and annotations to the consumer."""
        if self._frame and self._annotations and self._consumer.get():
            instance = self._create_instance(self._frame, self._annotations)
            self._frame = None
            self._annotations = None
            self._consumer.get().consume(instance)
            self._condition.notify_all()

    def _handle_eos(self) -> None:
        if self._eos:
            self._eos = False
            self._consumer.get().consume(None)
        else:
            self._eos = True

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