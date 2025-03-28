from abc import ABC, abstractmethod
from typing import List

from src.data.dataclasses.annotated_frame import AnnotatedFrame


class ClientContext(ABC):
    """An interface for client contexts."""
    @abstractmethod
    def put_batch(self, batch: List[AnnotatedFrame]) -> None:
        """
        Puts a batch of frames into the context.

        Args:
            batch (List[AnnotatedFrame]): The batch of frames to put.
        """
        raise NotImplementedError

    @abstractmethod
    def get_batch(self) -> List[AnnotatedFrame]:
        """
        Returns a batch of frames from the responses.

        Returns:
            List[AnnotatedFrame]: The batch of frames.
        """

        raise NotImplementedError
