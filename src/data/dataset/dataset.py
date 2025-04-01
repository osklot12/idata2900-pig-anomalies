from abc import ABC, abstractmethod
from typing import List

from src.data.dataclasses.annotated_frame import AnnotatedFrame
from src.data.dataset.dataset_split import DatasetSplit


class Dataset(ABC):
    """An interface for datasets."""

    @abstractmethod
    def get_shuffled_batch(self, split: DatasetSplit, batch_size: int) -> List[AnnotatedFrame]:
        """
        Samples a randomized batch of frame-annotation pairs from the specified dataset split.

        Args:
            split (DatasetSplit): the split to sample from
            batch_size (int): the batch size

        Returns:
            List[AnnotatedFrame]: the shuffled batch of frame-annotation pairs
        """
        raise NotImplementedError