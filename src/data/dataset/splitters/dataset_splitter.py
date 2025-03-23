from abc import ABC, abstractmethod
from typing import List, Iterable

from src.data.dataset_split import DatasetSplit


class DatasetSplitter(ABC):
    """An interface for dataset splitters."""

    @abstractmethod
    def update_dataset(self, new_dataset: Iterable[str]) -> None:
        """
        Updates the dataset.

        Args:
            new_dataset (Iterable[str]): an iterable of the id's of the updated dataset
        """
        raise NotImplementedError

    @abstractmethod
    def get_split(self, split: DatasetSplit) -> set[str]:
        """
        Returns the given dataset split.

        Args:
            split (DatasetSplit): the dataset split to get

        Returns:
            set[str]: a set of id's in the split
        """
        raise NotImplementedError