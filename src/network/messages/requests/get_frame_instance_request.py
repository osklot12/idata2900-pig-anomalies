from typing import List

from src.data.dataclasses.annotated_frame import AnnotatedFrame
from src.data.dataset_split import DatasetSplit
from src.network.messages.requests.request import Request, T
from src.network.messages.response.frame_instance_response import FrameInstanceResponse
from src.network.messages.response.response import Response
from src.network.server.server_context import ServerContext


class GetFrameInstanceRequest(Request[List[AnnotatedFrame]]):
    def __init__(self, split: DatasetSplit):
        """
        Initialize the request of a frame instance.

        Args:
            split (DatasetSplit): DatasetSplit of the frame instance.
        """
        self._split = split


    def execute(self, context: ServerContext) -> Response[T]:
        provider = context.get_frame_instance_provider()
        instance = provider.get_instance(self._split)
        return FrameInstanceResponse(instance)
