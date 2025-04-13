from typing import TypeVar, Generic

from src.data.dataset.streams.managed.factories.split_stream_factory import SplitStreamFactory
from src.network.messages.requests.handlers.request_handler import RequestHandler
from src.network.messages.requests.open_stream_request import OpenStreamRequest
from src.network.messages.responses.open_stream_response import OpenStreamResponse
from src.network.messages.responses.response import Response
from src.network.messages.responses.response_status import ResponseStatus
from src.network.server.session.session import Session

T = TypeVar("T")


class OpenStreamHandler(Generic[T], RequestHandler):
    """Handles OpenStreamRequest instances."""

    def __init__(self, session: Session[T], stream_factory: SplitStreamFactory):
        """
        Initializes an OpenStreamHandler instance.

        Args:
            session (Session[T]): network session for storing the stream
            stream_factory (SplitStreamFactory): factory for creating the stream
        """
        self._session = session
        self._stream_factory = stream_factory

    def handle(self, request: OpenStreamRequest) -> Response:
        response = OpenStreamResponse(ResponseStatus.ERROR)

        try:
            stream = self._stream_factory.create_stream(request.split)
            stream.run()
            self._session.set_stream(stream, request.split)
            response = OpenStreamResponse(ResponseStatus.SUCCESS)
        except RuntimeError as e:
            print(f"[OpenStreamHandler] Failed to create stream: {e}")

        return response