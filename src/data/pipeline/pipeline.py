from typing import Generic, TypeVar

from src.data.pipeline.component import Component
from src.data.pipeline.pipeline_builder import PipelineBuilder
from src.data.pipeline.pipeline_tail import PipelineTail

# input type of pipeline
A = TypeVar("A")

# current output type of pipeline
B = TypeVar("B")

# next output type
C = TypeVar("C")

class Pipeline(Generic[A, B, C], PipelineBuilder[A, B]):
    """Pipeline builder."""

    def __init__(self, head: Component[A, B]):
        """
        Initializes a Pipeline instance.

        Args:
            head (Component[A, B]): the first component of the pipeline
        """
        self._head = head

    def then(self, next_component: Component[B, C]) -> PipelineTail[A, C]:
        self._head.connect(next_component)
        return PipelineTail(head=self._head, tail=next_component)