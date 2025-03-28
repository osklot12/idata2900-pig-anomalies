from typing import List

from src.data.dataclasses.annotated_frame import AnnotatedFrame
from src.data.dataset_split import DatasetSplit
from src.data.frame_instance_provider import FrameInstanceProvider
from src.network.client.client_context import ClientContext
from src.network.client.client_network import NetworkClient
from src.network.messages.requests.get_frame_batch_request import GetFrameBatchRequest


class NetworkInstanceProvider(FrameInstanceProvider):


    def __init__(self, network_client: NetworkClient, context: ClientContext) -> None:
        self._net= network_client
        self._context = context

    def get_instance(self, split: DatasetSplit) -> AnnotatedFrame:
        pass

    def get_batch(self, split: DatasetSplit, batch_size: int) -> List[AnnotatedFrame]:
        request = GetFrameBatchRequest(split, batch_size)
        response = self._net.send_request(request)
        return response.execute(self._context)
