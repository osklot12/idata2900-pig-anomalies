from typing import List

from src.data.dataclasses.annotated_frame import AnnotatedFrame
from src.data.dataset_split import DatasetSplit
from src.data.frame_instance_provider import FrameInstanceProvider
from src.network.messages.requests.request import Request, S
from src.network.messages.responses.frame_batch_response import FrameBatchResponse


class GetFrameBatchRequest(Request[FrameInstanceProvider, List[AnnotatedFrame]]):
    """A request for getting a batch of annotated frames."""

    def __init__(self, split: DatasetSplit, batch_size: int):
        """
        Initializes the request.

        Args:
            split (DatasetSplit): the dataset split to sample from
            batch_size (int): the batch size
        """
        self._split = split
        self._batch_size = batch_size

    def execute(self, context: FrameInstanceProvider) -> FrameBatchResponse:
        batch = context.get_batch(self._split, self._batch_size)
        return FrameBatchResponse(batch)
