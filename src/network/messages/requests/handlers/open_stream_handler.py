from typing import Dict

from src.data.dataclasses.streamed_annotated_frame import StreamedAnnotatedFrame
from src.data.dataset.dataset_split import DatasetSplit
from src.data.dataset.streams.factories.managed_stream_factory import ManagedStreamFactory
from src.network.messages.requests.handlers.request_handler import RequestHandler
from src.network.messages.requests.open_stream_request import OpenStreamRequest
from src.network.messages.responses.open_stream_response import OpenStreamResponse
from src.network.messages.responses.response import Response
from src.network.messages.responses.response_status import ResponseStatus
from src.network.server.client_session import ClientSession


class OpenStreamHandler(RequestHandler):

    def __init__(self, session: ClientSession,
                 stream_factories: Dict[DatasetSplit, ManagedStreamFactory[StreamedAnnotatedFrame]]):
        self._session = session
        self._stream_factories = stream_factories

    def handle(self, request: OpenStreamRequest) -> Response:
        response = OpenStreamResponse(ResponseStatus.ERROR)

        try:
            print(f"Split: {request.split}")
            print(f"Stream factories: {self._stream_factories}")
            stream = self._stream_factories[request.split].create_stream()
            stream.start()
            self._session.streams[request.split] = stream
            response = OpenStreamResponse(ResponseStatus.SUCCESS)
        except RuntimeError as e:
            print(f"[OpenStreamHandler] Failed to create stream: {e}")

        return response