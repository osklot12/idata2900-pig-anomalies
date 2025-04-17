from typing import TypeVar, Iterable, Optional

from src.data.pipeline.base_component import BaseComponent
from src.data.pipeline.consumer import Consumer
from src.data.processing.processor import Processor

# input type
I = TypeVar("I")

# output type
O = TypeVar("O")

class SplittingPreprocessor(BaseComponent[I, O]):
    """A preprocessor that expands one input into multiple outputs."""

    def __init__(self, processor: Processor[I, Iterable[O]], consumer: Optional[Consumer[O]] = None):
        """
        Initializes a SplittingPreprocessor instance.

        Args:
            processor (Processor[I, Iterable[O]]): the processor to use
            consumer (Optional[Consumer[O]]): optional consumer of the data
        """
        super().__init__(consumer)
        self._processor: Processor[I, Iterable[O]] = processor

    def _consume(self, data: I, consumer: Consumer[O]) -> bool:
        return all(consumer.consume(d) for d in self._processor.process(data))