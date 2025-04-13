from typing import TypeVar

from src.network.messages.requests.get_batch_request import GetBatchRequest
from src.network.messages.requests.handlers.request_handler import RequestHandler
from src.network.messages.responses.get_batch_response import GetBatchResponse
from src.network.messages.responses.response import Response
from src.network.messages.responses.response_status import ResponseStatus
from src.network.server.session.session import Session

T = TypeVar("T")


class GetBatchHandler(RequestHandler):
    """Handles GetBatchRequest instances."""

    def __init__(self, session: Session[T]):
        """
        Initializes a GetBatchHandler instance.

        Args:
            session (Session[T]): the session to get the stream from
        """
        self._session = session

    def handle(self, request: GetBatchRequest) -> Response:
        response = GetBatchResponse(status=ResponseStatus.ERROR, batch=[])

        try:
            stream = self._session.get_stream(request.split)

            batch = []
            instance = stream.read()
            while instance is not None and len(batch) < request.batch_size:
                batch.append(instance)
                instance = stream.read()

            response = GetBatchResponse(status=ResponseStatus.SUCCESS, batch=batch)

        except RuntimeError as e:
            print(f"[GetBatchHandler] Failed to create batch: {e}")

        return response
