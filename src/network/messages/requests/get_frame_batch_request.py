from src.data.dataset_split import DatasetSplit
from src.network.messages.requests.request import Request
from src.network.messages.response.frame_batch_response import FrameBatchResponse
from src.network.messages.response.response import Response
from src.network.server.server_context import ServerContext


class GetFrameBatchRequest(Request):

    def __init__(self, split: DatasetSplit, batch_size: int):
        """
        Initializes the request.

        Args:
            split (DatasetSplit): The dataset split.
            batch_size (int): The batch size.
        """
        self._split = split
        self._batch_size = batch_size



    def execute(self, context: ServerContext) -> Response:
        provider = context.get_frame_instance_provider()
        batch = provider.get_batch(self._split, self._batch_size)
        return FrameBatchResponse(batch)