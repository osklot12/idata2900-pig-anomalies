from dataclasses import dataclass

from src.network.messages.responses.response import Response
from src.network.messages.responses.response_status import ResponseStatus


@dataclass(frozen=True)
class CloseStreamResponse(Response):
    """
    Response to a request to close a dataset stream.

    Attributes:
        status (ResponseStatus): the status of the response
    """
    status: ResponseStatus