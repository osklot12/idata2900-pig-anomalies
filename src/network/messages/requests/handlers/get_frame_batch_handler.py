from src.data.providers.instance_provider import InstanceProvider
from src.network.messages.requests.get_frame_batch_request import GetFrameBatchRequest
from src.network.messages.requests.handlers.request_handler import RequestHandler
from src.network.messages.responses.frame_batch_response import FrameBatchResponse


class GetFrameBatchHandler(RequestHandler):
    """A handler for GetFrameBatchRequest."""

    def __init__(self, instance_provider: InstanceProvider):
        """
        Initializes a GetFrameBatchHandler instance.

        Args:
            instance_provider (InstanceProvider): the frame instance provider
        """
        self._provider = instance_provider

    def handle(self, request: GetFrameBatchRequest) -> FrameBatchResponse:
        batch = self._provider.get_batch(request.split, request.batch_size)
        return FrameBatchResponse(batch)
