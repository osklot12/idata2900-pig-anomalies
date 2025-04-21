from abc import ABC, abstractmethod
from typing import TypeVar, Generic

from src.data.pipeline.pipeline_tail import PipelineTail

# data type for pipeline entry
A = TypeVar("A")

# data type for pipeline output
B = TypeVar("B")

class PipelineFactory(Generic[A, B], ABC):
    """Interface for pipeline factories."""

    @abstractmethod
    def create_pipeline(self) -> PipelineTail[A, B]:
        """
        Creates and returns a pipeline.

        Returns:
            PipelineTail[A, B]: the tail of the pipeline
        """
        raise NotImplementedError