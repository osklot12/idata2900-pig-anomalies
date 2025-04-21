from typing import TypeVar, Optional, Callable, Generic

from src.data.pipeline.consumer import Consumer
from src.data.structures.atomic_bool import AtomicBool

T = TypeVar("T")

class ConsumingRFunc(Generic[T], Consumer[T]):
    """Consumer adapter for a callback with a release."""

    def __init__(self, callback: Callable[[T, AtomicBool], bool], release: AtomicBool):
        """
        Initializes a ConsumingRFunc instance.

        Args:
            callback (Callable[[T, AtomicBool], bool]): callback that will be called
            release (AtomicBool): the release to use
        """
        self._callback = callback
        self._release = release

    def consume(self, data: Optional[T]) -> bool:
        return self._callback(data, self._release)