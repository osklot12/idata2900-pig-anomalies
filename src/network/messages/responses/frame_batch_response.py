from dataclasses import dataclass
from typing import List

from src.data.dataclasses.annotated_frame import AnnotatedFrame
from src.network.messages.responses.response import Response

@dataclass(frozen=True)
class FrameBatchResponse(Response):
    """
    A response containing a batch of annotated frames.

    Attributes:
        batch List[AnnotatedFrame]: the batch of annotated frames
    """
    batch: List[AnnotatedFrame]