from dataclasses import dataclass

from src.data.dataset.dataset_split import DatasetSplit
from src.network.messages.requests.request import Request


@dataclass(frozen=True)
class ReadStreamRequest(Request):
    """
    Request for reading a dataset stream.

    Attributes:
        split (DatasetSplit): the dataset split for stream to read from
    """
    split: DatasetSplit