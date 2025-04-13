from dataclasses import dataclass

from src.data.dataset.dataset_split import DatasetSplit
from src.network.messages.requests.request import Request


@dataclass(frozen=True)
class OpenStreamRequest(Request):
    """
    Request to open a dataset stream.

    Attributes:
        split (DatasetSplit): the dataset split to open stream for
    """
    split: DatasetSplit