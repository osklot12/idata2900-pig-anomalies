from typing import TypeVar, Optional

from src.data.pipeline.component import Component
from src.data.pipeline.consumer import Consumer
from src.data.pipeline.producer import T
from src.data.processing.processor import Processor
from src.data.structures.atomic_var import AtomicVar

I = TypeVar("I")
O = TypeVar("O")


class Preprocessor(Component[I, O]):
    """A Component wrapper for Processor instances."""

    def __init__(self, processor: Processor[I, O], consumer: Optional[Consumer[I]] = None):
        """
        Initializes a Preprocessor instance.

        Args:
            processor (Processor[I, O]): the processor to wrap
            consumer (Optional[Consumer[I]]): optional consumer of the processed data
        """
        self._processor = processor
        self._consumer = AtomicVar[Consumer[I]](consumer)

    def consume(self, data: Optional[T]) -> bool:
        success = False

        consumer = self._consumer.get()
        if consumer is not None:
            if data is not None:
                data = self._processor.process(data)

            success = consumer.consume(data)

        return success

    def connect(self, consumer: Consumer[T]) -> None:
        self._consumer = consumer