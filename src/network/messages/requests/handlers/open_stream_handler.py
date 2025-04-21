from typing import TypeVar, Generic, Dict

from src.data.dataclasses.compressed_annotated_frame import CompressedAnnotatedFrame
from src.data.dataset.dataset_split import DatasetSplit
from src.data.dataset.streams.managed.factories.manged_stream_factory import ManagedStreamFactory
from src.data.dataset.streams.managed.factories.split_stream_factory import SplitStreamFactory
from src.network.messages.requests.handlers.dataset_stream_factories import DatasetStreamFactories
from src.network.messages.requests.handlers.request_handler import RequestHandler
from src.network.messages.requests.open_stream_request import OpenStreamRequest
from src.network.messages.responses.open_stream_response import OpenStreamResponse
from src.network.messages.responses.response import Response
from src.network.messages.responses.response_status import ResponseStatus
from src.network.server.session.session import Session

T = TypeVar("T")


class OpenStreamHandler(Generic[T], RequestHandler[T]):
    """Handles OpenStreamRequest instances."""

    def __init__(self, session: Session[T], stream_factories: DatasetStreamFactories[T]):
        """
        Initializes an OpenStreamHandler instance.

        Args:
            session (Session[T]): network session for storing the stream
            stream_factories (DatasetStreamFactories[T]): factories for creating dataset streams
        """
        self._session = session
        self._stream_factories = stream_factories

    def handle(self, request: OpenStreamRequest) -> Response:
        response = OpenStreamResponse(ResponseStatus.ERROR)

        try:
            stream = self._stream_factories.for_split(request.split).create_stream()
            stream.run()
            self._session.set_stream(stream, request.split)
            response = OpenStreamResponse(ResponseStatus.SUCCESS)
        except Exception as e:
            print(f"[OpenStreamHandler] Failed to create stream: {e}")

        return response