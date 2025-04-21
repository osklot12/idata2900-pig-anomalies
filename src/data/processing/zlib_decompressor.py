import zlib
from typing import Optional

import numpy as np

from src.data.dataclasses.annotated_frame import AnnotatedFrame
from src.data.dataclasses.compressed_annotated_frame import CompressedAnnotatedFrame
from src.data.pipeline.consumer import Consumer
from src.data.processing.processor import Processor, I, O
from src.data.structures.atomic_var import AtomicVar


class ZlibDecompressor(Processor[CompressedAnnotatedFrame, AnnotatedFrame]):
    """Pipeline component for decompressing frames using zlib."""

    def __init__(self, consumer: Optional[Consumer[AnnotatedFrame]] = None):
        """
        Initializes a ZlibComponent instance.

        Args:
            consumer (Optional[Consumer[AnnotatedFrame]]): optional consumer of the data
        """
        self._consumer = AtomicVar[Consumer[AnnotatedFrame]](consumer)

    def process(self, data: CompressedAnnotatedFrame) -> AnnotatedFrame:
        return AnnotatedFrame(
            source=data.source,
            index=data.index,
            frame=np.frombuffer(zlib.decompress(data.frame), dtype=np.dtype(data.dtype)).reshape(data.shape),
            annotations=data.annotations
        )