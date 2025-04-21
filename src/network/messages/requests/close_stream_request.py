from dataclasses import dataclass

from src.data.dataset.dataset_split import DatasetSplit
from src.network.messages.requests.request import Request


@dataclass(frozen=True)
class CloseStreamRequest(Request):
    """
    Request to close a dataset stream.

    Attributes:
        split (DatasetSplit): the dataset split to close the stream for
    """
    split: DatasetSplit

    def __repr__(self):
        return f"CloseStreamRequest(split={self.split})"