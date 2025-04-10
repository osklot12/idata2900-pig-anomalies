from typing import List

from src.data.dataclasses.streamed_annotated_frame import StreamedAnnotatedFrame
from src.data.dataset.dataset_split import DatasetSplit
from src.network.client.network_client import NetworkClient
from src.network.messages.requests.get_batch_request import GetBatchRequest
from src.network.messages.requests.open_stream_request import OpenStreamRequest
from src.network.messages.responses.get_batch_response import GetBatchResponse
from src.network.messages.responses.open_stream_response import OpenStreamResponse
from src.network.messages.responses.response_status import ResponseStatus


class NetworkDatasetStream:

    def __init__(self, client: NetworkClient, split: DatasetSplit):
        self._client = client
        self._split = split

        response = self._client.send_request(OpenStreamRequest(self._split))
        if not isinstance(response, OpenStreamResponse):
            raise RuntimeError("Got unexpected response from server")

        if not response.status == ResponseStatus.SUCCESS:
            raise RuntimeError("Could not open stream")

    def get_batch(self, batch_size: int) -> List[StreamedAnnotatedFrame]:
        response = self._client.send_request(GetBatchRequest(split=self._split, batch_size=batch_size))
        if not isinstance(response, GetBatchResponse):
            raise RuntimeError("Got unexpected response from server")

        if not response.status == ResponseStatus.SUCCESS:
            raise RuntimeError("Could not get batch")

        return response.batch
