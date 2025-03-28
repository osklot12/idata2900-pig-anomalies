from typing import List

from src.data.dataclasses.annotated_frame import AnnotatedFrame
from src.data.dataset_split import DatasetSplit
from src.network.messages.requests.request import Request, T
from src.network.messages.response.frame_batch_response import FrameBatchResponse
from src.network.messages.response.response import Response
from src.network.server.server_context import ServerContext


class GetFrameBatchRequest(Request[List[AnnotatedFrame]]):
    def __init__(self, split: DatasetSplit, batch_size: int):
        """
        Initializes the request.

        Args:
            split (DatasetSplit): The dataset split.
            batch_size (int): The batch size.
        """
        self._split = split
        self._batch_size = batch_size


    def execute(self, context: object ) -> Response[T]:
        provider = context.get_frame_instance_provider()
        batch = provider.get_batch(self._split, self._batch_size)
        return FrameBatchResponse(batch)
