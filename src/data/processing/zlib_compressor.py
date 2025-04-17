import zlib
from typing import Optional

from src.data.dataclasses.annotated_frame import AnnotatedFrame
from src.data.dataclasses.compressed_annotated_frame import CompressedAnnotatedFrame
from src.data.pipeline.consumer import Consumer
from src.data.processing.processor import Processor
from src.data.structures.atomic_var import AtomicVar


class ZlibCompressor(Processor[AnnotatedFrame, CompressedAnnotatedFrame]):
    """Pipeline component for compressing frames using zlib."""

    def __init__(self, consumer: Optional[Consumer[CompressedAnnotatedFrame]] = None):
        """
        Initializes a ZlibComponent instance.

        Args:
            consumer (Optional[Consumer[CompressedAnnotatedFrame]]): optional consumer of the data
        """
        self._consumer = AtomicVar[Consumer[CompressedAnnotatedFrame]](consumer)

    def process(self, data: AnnotatedFrame) -> CompressedAnnotatedFrame:
        frame = data.frame
        return CompressedAnnotatedFrame(
            source=data.source,
            index=data.index,
            frame=zlib.compress(frame),
            shape=frame.shape,
            dtype=str(frame.dtype),
            annotations=data.annotations,
        )