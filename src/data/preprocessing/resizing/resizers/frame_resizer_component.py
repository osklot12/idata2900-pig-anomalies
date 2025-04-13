from typing import Optional

from src.data.dataclasses.frame import Frame
from src.data.dataclasses.streamed_annotated_frame import StreamedAnnotatedFrame
from src.data.pipeline.component import Component
from src.data.pipeline.consumer import Consumer
from src.data.preprocessing.resizing.resizers.frame_resizer import FrameResizer
from src.data.structures.atomic_var import AtomicVar


class FrameResizerComponent(Component[StreamedAnnotatedFrame]):
    """Pipeline component adapter for frame resizers."""

    def __init__(self, resizer: FrameResizer, consumer: Optional[Consumer[StreamedAnnotatedFrame]] = None):
        """
        Initializes a FrameResizerComponent instance.

        Args:
            resizer (FrameResizer): the resizer for resizing frames
            consumer (Optional[Consumer[Frame]]): the consumer of the streamed frames
        """
        self._resizer = resizer
        self._consumer = AtomicVar[Consumer[StreamedAnnotatedFrame]](consumer)

    def consume(self, data: Optional[StreamedAnnotatedFrame]) -> bool:
        resized = None

        if data is not None:
            resized = self._resize_frame(data)

        return self._consumer.get().consume(resized)

    def connect(self, consumer: Consumer[StreamedAnnotatedFrame]) -> None:
        self._consumer.set(consumer)

    def _resize_frame(self, instance: StreamedAnnotatedFrame) -> StreamedAnnotatedFrame:
        return StreamedAnnotatedFrame(
            source=instance.source,
            index=instance.index,
            frame=self._resizer.resize_frame(instance.frame),
            annotations=instance.annotations
        )