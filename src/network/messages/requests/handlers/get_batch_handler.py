from src.network.messages.requests.get_batch_request import GetBatchRequest
from src.network.messages.requests.handlers.request_handler import RequestHandler
from src.network.messages.responses.get_batch_response import GetBatchResponse
from src.network.messages.responses.response import Response
from src.network.messages.responses.response_status import ResponseStatus
from src.network.server.client_session import ClientSession


class GetBatchHandler(RequestHandler):

    def __init__(self, session: ClientSession):
        self._session = session

    def handle(self, request: GetBatchRequest) -> Response:
        response = GetBatchResponse(status=ResponseStatus.ERROR, batch=[])

        try:
            stream = self._session.streams[request.split].stream
            batch = []
            i = 0

            instance = stream.read()
            while i < request.batch_size and instance:
                batch.append(instance)
                instance = stream.read()

                i += 1

            response = GetBatchResponse(status=ResponseStatus.SUCCESS, batch=batch)

        except RuntimeError as e:
            print(f"[GetBatchHandler] Failed to create batch: {e}")

        return response