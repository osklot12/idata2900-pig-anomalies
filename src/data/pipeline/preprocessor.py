from typing import TypeVar, Optional, Generic

from src.data.pipeline.base_component import BaseComponent
from src.data.pipeline.consumer import Consumer
from src.data.processing.processor import Processor

# input data
I = TypeVar("I")

# output data
O = TypeVar("O")


class Preprocessor(Generic[I, O], BaseComponent[I, O]):
    """A Component wrapper for Processor instances."""

    def __init__(self, processor: Processor[I, O], consumer: Optional[Consumer[O]] = None):
        """
        Initializes a Preprocessor instance.

        Args:
            processor (Processor[I, O]): the processor to wrap
            consumer (Optional[Consumer[I]]): optional consumer of the processed data
        """
        super().__init__(consumer)
        self._processor = processor

    def _consume(self, data: I, consumer: Consumer[O]) -> bool:
        return consumer.consume(self._processor.process(data))