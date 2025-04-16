from abc import ABC, abstractmethod
from typing import TypeVar, Generic

from src.data.pipeline.component import Component
from src.data.pipeline.pipeline_tail import PipelineTail

# input type of pipeline
A = TypeVar("A")

# current output type of pipeline
B = TypeVar("B")

# next output type
C = TypeVar("C")


class PipelineBuilder(Generic[A, B], ABC):
    """Connects a pipeline component to the given next component."""

    @abstractmethod
    def then(self, next_component: Component[B, C]) -> PipelineTail[A, C]:
        """
        Adds a component to the current pipeline and returns a builder for the extended pipeline.

        Args:
            next_component (Component[A, C]): the next component of the pipeline

        Returns:
            PipelineTail[A, C]: builder representing the new end of the pipeline
        """
        raise NotImplementedError