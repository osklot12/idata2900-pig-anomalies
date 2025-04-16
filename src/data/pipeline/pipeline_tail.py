from typing import TypeVar, Generic

from src.data.pipeline.component import Component
from src.data.pipeline.consumer import Consumer
from src.data.pipeline.producer import Producer

A = TypeVar("A")
B = TypeVar("B")
C = TypeVar("C")

class PipelineTail(Generic[A, B]):
    """Tail for a pipeline builder."""

    def __init__(self, head: Consumer[A], tail: Producer[B]):
        self._head: Consumer[A] = head
        self._tail: Producer[B] = tail

    def then(self, next_component: Component[B, C]) -> "PipelineTail[A, C]":
        self._tail.connect(next_component)
        return PipelineTail(head=self._head, tail=next_component)

    def into(self, sink: Consumer[B]) -> "Consumer[A]":
        self._tail.connect(sink)
        return self._head