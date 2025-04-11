from typing import Optional

from src.data.dataclasses.frame import Frame
from src.data.pipeline.consumer import Consumer
from src.data.pipeline.producer import Producer
from src.data.preprocessing.resizing.resizers.frame_resizer import FrameResizer
from src.data.structures.atomic_var import AtomicVar


class FrameResizerComponent(Consumer[Frame], Producer[Frame]):
    """Pipeline component adapter for frame resizers."""

    def __init__(self, resizer: FrameResizer, consumer: Optional[Consumer[Frame]] = None):
        """
        Initializes a FrameResizerComponent instance.

        Args:
            resizer (FrameResizer): the resizer for resizing frames
            consumer (Optional[Consumer[Frame]]): the consumer of the streamed frames
        """
        self._resizer = resizer
        self._consumer = AtomicVar[Consumer[Frame]](consumer)

    def consume(self, data: Optional[Frame]) -> bool:
        resized = self._resize_frame(data)
        return self._consumer.get().consume(resized)

    def set_consumer(self, consumer: Consumer[Frame]) -> None:
        self._consumer.set(consumer)

    def _resize_frame(self, frame: Frame) -> Frame:
        return Frame(
            frame.source,
            frame.index,
            self._resizer.resize_frame(frame.data)
        )