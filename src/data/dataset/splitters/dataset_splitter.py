from abc import ABC, abstractmethod
from typing import List

from src.data.dataset_split import DatasetSplit


class DatasetSplitter(ABC):
    """An interface for dataset splitters."""

    @abstractmethod
    def update_dataset(self, new_dataset: List[str]) -> None:
        """
        Updates the dataset.

        Args:
            new_dataset (List[str]): a list of the id's of the updated dataset
        """
        raise NotImplementedError

    @abstractmethod
    def get_split(self, split: DatasetSplit) -> List[str]:
        """
        Returns the given dataset split.

        Args:
            split (DatasetSplit): the dataset split

        Returns:
            List[str]: a list of id's in the split
        """
        raise NotImplementedError