from abc import ABC, abstractmethod
from typing import List, Iterable, Optional

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

    @abstractmethod
    def get_split_ratio(self, split: DatasetSplit) -> float:
        """
        Returns the ratio for the given dataset split.

        Args:
            split (DatasetSplit): the dataset split to get ratio for

        Returns:
            float: the ratio for the split
        """
        raise NotImplementedError

    @abstractmethod
    def get_split_for_id(self, id_: str) -> Optional[DatasetSplit]:
        """
        Returns the dataset split for the given id.

        Args:
            id_ (str): the id to get the split for

        Returns:
            DatasetSplit: the dataset split for the given id, or None if id is not recognized
        """
        raise NotImplementedError