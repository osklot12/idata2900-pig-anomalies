from dataclasses import dataclass
from typing import TypeVar, Generic, Optional

from src.network.messages.responses.response import Response
from src.network.messages.responses.response_status import ResponseStatus

T = TypeVar("T")


@dataclass(frozen=True)
class ReadStreamResponse(Generic[T], Response):
    """
    Response to a ReadStreamRequest.

    Attributes:
        status (ResponseStatus): the status of the response
        instance (Optional[T]): the read stream instance, or None if end of stream or error
    """
    status: ResponseStatus
    instance: Optional[T]