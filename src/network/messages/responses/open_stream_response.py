from dataclasses import dataclass

from src.network.messages.responses.response import Response
from src.network.messages.responses.response_status import ResponseStatus


@dataclass(frozen=True)
class OpenStreamResponse(Response):
    """
    Response to a request to open a stream.

    Attributes:
        status (ResponseStatus): the status of the response
    """
    status: ResponseStatus