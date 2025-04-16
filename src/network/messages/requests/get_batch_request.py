from dataclasses import dataclass

from src.data.dataset.dataset_split import DatasetSplit
from src.network.messages.requests.request import Request


@dataclass(frozen=True)
class GetBatchRequest(Request):
    """
    Request to get a batch of data.

    Attributes:
        split (DatasetSplit): the dataset split to get the data from
        batch_size (int): the batch size
    """
    split: DatasetSplit
    batch_size: int

    def __repr__(self):
        return f"GetBatchRequest(split={self.split}, batch_size={self.batch_size})"