from typing import List

from src.data.dataclasses.annotated_frame import AnnotatedFrame
from src.data.dataset.dataset_split import DatasetSplit
from src.data.frame_instance_provider import FrameInstanceProvider
from src.network.client.network_client import NetworkClient
from src.network.messages.requests.get_frame_batch_request import GetFrameBatchRequest
from src.network.messages.responses.frame_batch_response import FrameBatchResponse


class NetworkFrameInstanceProvider(FrameInstanceProvider):
    """Provides annotated frame instances from a network server."""

    def __init__(self, server_ip: str, network_client: NetworkClient):
        """
        Initializes a NetworkFrameInstanceProvider.

        Args:
            server_ip (str): the IP address of the network server
            network_client (SimpleNetworkClient): the network client
        """
        self._server_ip = server_ip
        self._client = network_client
        self._client.connect(server_ip)

    def get_batch(self, split: DatasetSplit, batch_size: int) -> List[AnnotatedFrame]:
        response = self._client.send_request(GetFrameBatchRequest(split, batch_size))

        if not isinstance(response, FrameBatchResponse):
            raise RuntimeError(f"Unexpected response type {type(response)}")

        return response.batch