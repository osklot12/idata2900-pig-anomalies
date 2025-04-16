import zlib
from typing import Optional

from src.data.dataclasses.annotated_frame import AnnotatedFrame
from src.data.dataclasses.compressed_annotated_frame import CompressedAnnotatedFrame
from src.data.pipeline.component import Component
from src.data.pipeline.consumer import Consumer
from src.data.structures.atomic_var import AtomicVar


class ZlibCompressor(Component[AnnotatedFrame, CompressedAnnotatedFrame]):
    """Pipeline component for compressing frames using zlib."""

    def __init__(self, consumer: Optional[Consumer[CompressedAnnotatedFrame]] = None):
        """
        Initializes a ZlibComponent instance.

        Args:
            consumer (Optional[Consumer[CompressedAnnotatedFrame]]): optional consumer of the data
        """
        self._consumer = AtomicVar[Consumer[CompressedAnnotatedFrame]](consumer)

    def consume(self, data: Optional[AnnotatedFrame]) -> bool:
        frame = data.frame
        return self._consumer.get().consume(
            CompressedAnnotatedFrame(
                source=data.source,
                index=data.index,
                frame=zlib.compress(frame),
                shape=frame.shape,
                dtype=str(frame.dtype),
                annotations=data.annotations,
            )
        )

    def connect(self, consumer: Consumer[CompressedAnnotatedFrame]) -> None:
        self._consumer.set(consumer)