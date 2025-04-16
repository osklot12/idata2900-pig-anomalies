from typing import TypeVar, Optional

from src.data.pipeline.component import Component
from src.data.pipeline.consumer import Consumer
from src.data.processing.processor import Processor
from src.data.structures.atomic_var import AtomicVar

# input data
I = TypeVar("I")

# output data
O = TypeVar("O")


class Preprocessor(Component[I, O]):
    """A Component wrapper for Processor instances."""

    def __init__(self, processor: Processor[I, O], consumer: Optional[Consumer[O]] = None):
        """
        Initializes a Preprocessor instance.

        Args:
            processor (Processor[I, O]): the processor to wrap
            consumer (Optional[Consumer[I]]): optional consumer of the processed data
        """
        self._processor = processor
        self._consumer = AtomicVar[Consumer[O]](consumer)

    def consume(self, data: Optional[I]) -> bool:
        success = False

        consumer = self._consumer.get()
        if consumer is not None:
            if data is not None:
                data = self._processor.process(data)

            success = consumer.consume(data)

        return success

    def connect(self, consumer: Consumer[O]) -> None:
        self._consumer.set(consumer)