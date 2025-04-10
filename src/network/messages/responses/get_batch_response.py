from dataclasses import dataclass
from typing import List

from src.data.dataclasses.streamed_annotated_frame import StreamedAnnotatedFrame
from src.network.messages.responses.response import Response
from src.network.messages.responses.response_status import ResponseStatus


@dataclass(frozen=True)
class GetBatchResponse(Response):
    status: ResponseStatus
    batch: List[StreamedAnnotatedFrame]