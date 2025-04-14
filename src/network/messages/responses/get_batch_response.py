from dataclasses import dataclass
from typing import List, TypeVar, Generic

from src.network.messages.responses.response import Response
from src.network.messages.responses.response_status import ResponseStatus

T = TypeVar("T")

@dataclass(frozen=True)
class GetBatchResponse(Generic[T], Response):
    """
    A response to a request to get a batch of data.

    Attributes:
        status (ResponseStatus): the status of the response
        batch (List[T]): the batch of data
    """
    status: ResponseStatus
    batch: List[T]