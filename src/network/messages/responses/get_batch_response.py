from dataclasses import dataclass
from typing import List, TypeVar

from src.data.dataclasses.streamed_annotated_frame import StreamedAnnotatedFrame
from src.network.messages.responses.response import Response
from src.network.messages.responses.response_status import ResponseStatus

T = TypeVar("T")

@dataclass(frozen=True)
class GetBatchResponse(Generic[T], Response):
    status: ResponseStatus
    batch: List[StreamedAnnotatedFrame]