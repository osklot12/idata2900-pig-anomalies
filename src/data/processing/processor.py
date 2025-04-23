from abc import ABC, abstractmethod
from typing import TypeVar, Generic

I = TypeVar("I")
O = TypeVar("O")

class Processor(Generic[I, O], ABC):
    """Interface for data processors."""

    @abstractmethod
    def process(self, data: I) -> O:
        """
        Processes the given data.

        Args:
            data (I): the data to process

        Returns:
            O: the processed data
        """
        raise NotImplementedError