from typing import List

from src.data.dataclasses.annotated_frame import AnnotatedFrame
from src.data.dataset.dataset_split import DatasetSplit
from src.data.errors.data_retrieval_error import DataRetrievalError
from src.data.providers.frame_instance_provider import FrameInstanceProvider
from src.network.client.network_client import NetworkClient
from src.network.messages.requests.get_frame_batch_request import GetFrameBatchRequest
from src.network.messages.responses.frame_batch_response import FrameBatchResponse


class NetworkFrameInstanceProvider(FrameInstanceProvider):
    """Provides annotated frame instances from a network server."""

    def __init__(self, network_client: NetworkClient):
        """
        Initializes a NetworkFrameInstanceProvider.

        Args:
            network_client (SimpleNetworkClient): the network client
        """
        self._client = network_client

    def get_batch(self, split: DatasetSplit, batch_size: int) -> List[AnnotatedFrame]:
        try:
            response = self._client.send_request(GetFrameBatchRequest(split, batch_size))
        except Exception as e:
            raise DataRetrievalError(f"Failed to send request: {e}", e)

        if not isinstance(response, FrameBatchResponse):
            raise DataRetrievalError(f"Unexpected response type {type(response)}")

        return response.batch
