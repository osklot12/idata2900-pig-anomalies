from typing import TypeVar, Optional, Generic

from src.data.dataset.dataset_split import DatasetSplit
from src.data.dataset.streams.closable_stream import ClosableStream
from src.network.client.network_client import NetworkClient
from src.network.messages.requests.close_stream_request import CloseStreamRequest
from src.network.messages.requests.open_stream_request import OpenStreamRequest
from src.network.messages.requests.read_stream_request import ReadStreamRequest
from src.network.messages.responses.close_stream_response import CloseStreamResponse
from src.network.messages.responses.open_stream_response import OpenStreamResponse
from src.network.messages.responses.read_stream_response import ReadStreamResponse
from src.network.messages.responses.response_status import ResponseStatus

# stream output data type
T = TypeVar("T")


class NetworkStream(Generic[T], ClosableStream[T]):
    """Dataset stream that fetches data from a server."""

    def __init__(self, client: NetworkClient, split: DatasetSplit, data_type: type[T]):
        """
        Initializes a NetworkStream instance.

        Args:
            client (NetworkClient): network client for sending requests to server
            split (DatasetSplit): dataset split to get data from
            data_type (type[T]): data type
        """
        self._client = client
        self._split = split
        self._data_type = data_type

        self._stream_open = False

    def read(self) -> Optional[T]:
        if not self._stream_open:
            self._open_stream()
            self._stream_open = True

        request = ReadStreamRequest(split=self._split)
        response = self._client.send_request(request)

        if not isinstance(response, ReadStreamResponse):
            raise RuntimeError("Got unexpected response from server")

        if not response.status == ResponseStatus.SUCCESS:
            raise RuntimeError("Could not get batch")

        if response.instance is not None and not isinstance(response.instance, self._data_type):
            raise RuntimeError("Response contains unexpected data type")

        return response.instance

    def _open_stream(self) -> None:
        """Opens the stream."""
        request = OpenStreamRequest(split=self._split)
        response = self._client.send_request(request)

        if not isinstance(response, OpenStreamResponse):
            raise RuntimeError("Got unexpected response from server")

        if not response.status == ResponseStatus.SUCCESS:
            raise RuntimeError(f"Could not open stream for split: {self._split}")

    def close(self) -> None:
        request = CloseStreamRequest(split=self._split)
        response = self._client.send_request(request)

        if not isinstance(response, CloseStreamResponse):
            raise RuntimeError("Got unexpected response from server")

        if response.status != ResponseStatus.SUCCESS:
            raise RuntimeError("Could not close stream")