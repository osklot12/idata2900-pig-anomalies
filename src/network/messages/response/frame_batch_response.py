from typing import List

from src.data.dataclasses.annotated_frame import AnnotatedFrame
from src.network.client.client_context import ClientContext
from src.network.messages.response.response import Response, T
from src.network.server.server_context import ServerContext


class FrameBatchResponse(Response[List[AnnotatedFrame]]):
    def __init__(self, batch: List[AnnotatedFrame]):
        """
        Initialize a FrameBatchResponse object.

        Args:
            batch (List[AnnotatedFrame]): List of frames.
        """
        self._batch = batch

    def execute(self, context: ClientContext) -> T:
        return self._batch