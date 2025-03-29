from dataclasses import dataclass

from src.data.dataset.dataset_split import DatasetSplit
from src.network.messages.requests.request import Request


@dataclass(frozen=True)
class GetFrameBatchRequest(Request):
    """
    A request to get a batch of annotated frames.

    Attributes:
        split (DatasetSplit): the dataset split to sample from
        batch_size (int): the batch size
    """
    split: DatasetSplit
    batch_size: int
