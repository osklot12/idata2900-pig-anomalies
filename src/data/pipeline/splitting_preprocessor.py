from typing import TypeVar, Iterable

from src.data.pipeline.component import Component
from src.data.processing.processor import Processor

# input type
I = TypeVar("I")

# output type
O = TypeVar("O")

class SplittingPreprocessor(Component[I, O]):
    """A preprocessor that expands one input into multiple outputs."""

    def __init__(self, processor: Processor[I, Iterable[O]]):