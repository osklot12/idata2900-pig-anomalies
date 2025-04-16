from typing import TypeVar, Generic

from src.data.pipeline.component import Component
from src.data.pipeline.consumer import Consumer
from src.data.pipeline.producer import Producer

# input type of pipeline
A = TypeVar("A")

# current output type of pipeline
B = TypeVar("B")

# next output type
C = TypeVar("C")

class PipelineTail(Generic[A, B]):
    """Tail for a pipeline builder."""

    def __init__(self, head: Consumer[A], tail: Producer[B]):
        self._head: Consumer[A] = head
        self._tail: Producer[B] = tail

    def then(self, next_component: Component[B, C]) -> "PipelineTail[A, C]":
        """
        Adds a component to the current pipeline and returns a builder for the extended pipeline.

        Args:
            next_component (Component[A, C]): the next component of the pipeline

        Returns:
            PipelineTail[A, C]: builder representing the new end of the pipeline
        """
        self._tail.connect(next_component)
        return PipelineTail(head=self._head, tail=next_component)

    def into(self, sink: Consumer[B]) -> "Consumer[A]":
        """
        Connects the final consumer and builds the pipeline.

        Args:
            sink (Consumer[B]): final consumer (sink) of the pipeline

        Returns:
            Consumer[A]: the first component of the pipeline
        """
        self._tail.connect(sink)
        return self._head