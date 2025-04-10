from dataclasses import dataclass
from typing import Dict, Optional

from src.data.dataclasses.streamed_annotated_frame import StreamedAnnotatedFrame
from src.data.dataset.dataset_split import DatasetSplit
from src.data.dataset.streams.managed_stream import ManagedStream


@dataclass(frozen=True)
class ClientSession:
    """Network session for a client."""
    streams: Dict[DatasetSplit, Optional[ManagedStream[StreamedAnnotatedFrame]]]
