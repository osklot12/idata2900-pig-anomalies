from dataclasses import dataclass

from src.network.messages.responses.response import Response
from src.network.messages.responses.response_status import ResponseStatus


@dataclass(frozen=True)
class OpenStreamResponse(Response):
    status: ResponseStatus