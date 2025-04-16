from typing import Generic, TypeVar

from src.data.pipeline.component import Component
from src.data.pipeline.pipeline_tail import PipelineTail
from src.data.pipeline.producer import Producer

A = TypeVar("A")
B = TypeVar("B")
C = TypeVar("C")

class Pipeline(Generic[A, B, C]):
    """Pipeline builder class."""

    def __init__(self, head: Component[A, B]):
        self._head = head

    def then(self, next_component: Component[B, C]) -> PipelineTail[A, C]:
        self._head.connect(next_component)
        return PipelineTail(head=self._head, tail=next_component)