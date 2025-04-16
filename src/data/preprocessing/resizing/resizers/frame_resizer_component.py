from typing import Optional

from src.data.dataclasses.frame import Frame
from src.data.dataclasses.annotated_frame import AnnotatedFrame
from src.data.pipeline.component import Component
from src.data.pipeline.consumer import Consumer
from src.data.processing.frame_resizer import FrameResizer
from src.data.structures.atomic_var import AtomicVar


class FrameResizerComponent(Component[AnnotatedFrame, AnnotatedFrame]):
    """Pipeline component adapter for frame resizers."""

    def __init__(self, resizer: FrameResizer, consumer: Optional[Consumer[AnnotatedFrame]] = None):
        """
        Initializes a FrameResizerComponent instance.

        Args:
            resizer (FrameResizer): the resizer for resizing frames
            consumer (Optional[Consumer[Frame]]): the consumer of the streamed frames
        """
        self._resizer = resizer
        self._consumer = AtomicVar[Consumer[AnnotatedFrame]](consumer)

    def consume(self, data: Optional[AnnotatedFrame]) -> bool:
        resized = None

        if data is not None:
            resized = self._resize_frame(data)

        return self._consumer.get().consume(resized)

    def connect(self, consumer: Consumer[AnnotatedFrame]) -> None:
        self._consumer.set(consumer)

    def _resize_frame(self, instance: AnnotatedFrame) -> AnnotatedFrame:
        return AnnotatedFrame(
            source=instance.source,
            index=instance.index,
            frame=self._resizer.resize_frame(instance.frame),
            annotations=instance.annotations
        )