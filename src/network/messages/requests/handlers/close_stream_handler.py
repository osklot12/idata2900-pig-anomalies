from typing import TypeVar, Generic

from src.network.messages.requests.close_stream_request import CloseStreamRequest
from src.network.messages.requests.handlers.request_handler import RequestHandler
from src.network.messages.responses.close_stream_response import CloseStreamResponse
from src.network.messages.responses.response import Response
from src.network.messages.responses.response_status import ResponseStatus
from src.network.server.session.session import Session

T = TypeVar("T")

class CloseStreamHandler(Generic[T], RequestHandler[T]):
    """Handles CloseStreamRequest instances."""

    def __init__(self, session: Session[T]):
        """
        Initializes a CloseStreamHandler instance.

        Args:
            session (Session[T]): the session close the stream for
        """
        self._session = session

    def handle(self, request: CloseStreamRequest) -> Response:
        response = CloseStreamResponse(status=ResponseStatus.ERROR)

        try:
            self._session.set_stream(stream=None, split=request.split)
            response = CloseStreamResponse(status=ResponseStatus.SUCCESS)

        except RuntimeError as e:
            print(f"[CloseStreamHandler] Failed to close stream: {e}")

        return response