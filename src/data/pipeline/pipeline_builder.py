from abc import ABC, abstractmethod
from typing import TypeVar, Generic

from src.data.pipeline.consumer import Consumer

# pipeline input data type
A = TypeVar("A")

# sink input data type
B = TypeVar("B")

class PipelineBuilder(Generic[A, B], ABC):
    """Interface for pipeline builders."""

    @abstractmethod
    def into(self, sink: Consumer[B]) -> Consumer[A]:
        """
        Connects the final consumer and builds the pipeline.

        Args:
            sink (Consumer[B]): final consumer (sink) of the pipeline

        Returns:
            Consumer[A]: the first component of the pipeline
        """
        raise NotImplementedError