from typing import TypeVar, Generic

from src.network.messages.requests.handlers.request_handler import RequestHandler
from src.network.messages.requests.read_stream_request import ReadStreamRequest
from src.network.messages.responses.read_stream_response import ReadStreamResponse
from src.network.messages.responses.response import Response
from src.network.messages.responses.response_status import ResponseStatus
from src.network.server.session.session import Session

T = TypeVar("T")


class ReadStreamHandler(Generic[T], RequestHandler[ReadStreamRequest]):
    """Handles ReadStreamRequest instances."""

    def __init__(self, session: Session[T]):
        """
        Initializes a ReadStreamHandler instance.

        Args:
            session (Session[T]): the session to get the stream from
        """
        self._session = session

    def handle(self, request: ReadStreamRequest) -> Response:
        status = ResponseStatus.ERROR
        instance = None

        try:
            stream = self._session.get_stream(request.split)
            if stream is None:
                raise RuntimeError(f"No stream open for {request.split}")

            instance = stream.read()
            status = ResponseStatus.SUCCESS

        except Exception as e:
            print(f"[ReadStreamHandler] Failed to read stream: {e}")

        return ReadStreamResponse(status=status, instance=instance)
