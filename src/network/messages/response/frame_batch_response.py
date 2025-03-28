from typing import List

from src.data.dataclasses.annotated_frame import AnnotatedFrame
from src.network.messages.response.response import Response, R


class FrameBatchResponse(Response[List[AnnotatedFrame]]):
    """A response for getting a batch of annotated frames."""

    def __init__(self, batch: List[AnnotatedFrame]):
        """
        Initialize a FrameBatchResponse instance.

        Args:
            batch (List[AnnotatedFrame]): a list of annotated frames
        """
        self._batch = batch

    def execute(self) -> List[AnnotatedFrame]:
        return self._batch