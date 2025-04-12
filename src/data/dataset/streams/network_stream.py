from typing import TypeVar, Optional, Generic, List

from src.data.dataset.dataset_split import DatasetSplit
from src.data.dataset.streams.stream import Stream
from src.network.client.network_client import NetworkClient
from src.network.messages.requests.get_batch_request import GetBatchRequest
from src.network.messages.requests.open_stream_request import OpenStreamRequest
from src.network.messages.responses.get_batch_response import GetBatchResponse
from src.network.messages.responses.open_stream_response import OpenStreamResponse
from src.network.messages.responses.response_status import ResponseStatus

T = TypeVar("T")


class NetworkStream(Generic[T], Stream[List[T]]):
    """Dataset stream that fetches data from a server."""

    def __init__(self, client: NetworkClient, split: DatasetSplit, batch_type: type[T], batch_size: int = 8):
        """
        Initializes a NetworkStream instance.

        Args:
            client (NetworkClient): network client for sending requests to server
            split (DatasetSplit): dataset split to get data from
            batch_type (type[T]): data type contained by the batches
            batch_size (int): the size of the batches to request, defaults to 8
        """
        self._client = client
        self._split = split
        self._batch_type = batch_type
        self._batch_size = batch_size

        response = self._client.send_request(OpenStreamRequest(self._split))
        if not isinstance(response, OpenStreamResponse):
            raise RuntimeError("Got unexpected response from server")

        if not response.status == ResponseStatus.SUCCESS:
            raise RuntimeError("Could not open stream")

    def read(self) -> Optional[List[T]]:
        response = self._client.send_request(GetBatchRequest(split=self._split, batch_size=self._batch_size))
        if not isinstance(response, GetBatchResponse):
            raise RuntimeError("Got unexpected response from server")

        if not response.status == ResponseStatus.SUCCESS:
            raise RuntimeError("Could not get batch")

        if not all(isinstance(item, self._batch_type) for item in response.batch):
            raise RuntimeError("Batch contains unexpected item types")

        return response.batch