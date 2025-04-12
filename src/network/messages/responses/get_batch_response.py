from dataclasses import dataclass
from typing import List, TypeVar, Generic

from src.network.messages.responses.response import Response
from src.network.messages.responses.response_status import ResponseStatus

T = TypeVar("T")

@dataclass(frozen=True)
class GetBatchResponse(Generic[T], Response):
    status: ResponseStatus
    batch: List[T]