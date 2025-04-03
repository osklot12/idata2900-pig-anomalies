import hashlib
from typing import Dict, Iterable, Optional

from src.data.dataset.splitters.dataset_splitter import DatasetSplitter
from src.data.dataset.dataset_split import DatasetSplit


class ConsistentDatasetSplitter(DatasetSplitter):
    """A consistent dataset splitter, always putting the same ID's in the same split."""

    def __init__(self, dataset_ids: Iterable[str] = None, train_ratio: float = 0.8, val_ratio: float = 0.1,
                 seed: int = 42):
        """
        Initializes a ConsistentDatasetSplitter instance.

        Args:
            dataset_ids (Iterable[str]): optional iterable of dataset id's to update the dataset upon construction
            train_ratio (float): the ratio of the training set size
            val_ratio (float): the ratio of the validation set size
            seed (int): the random seed
        """
        if train_ratio + val_ratio < 0 or train_ratio + val_ratio > 1:
            raise ValueError("The sum of train_ratio and val_ratio must be in the interval [0, 1]")

        self._train_threshold = train_ratio
        self._val_threshold = val_ratio
        self._seed = seed

        self._dataset: set[str] = set()
        self._train_split: set[str] = set()
        self._val_split: set[str] = set()
        self._test_split: set[str] = set()

        self._split_map: Dict[DatasetSplit, set[str]] = {
            DatasetSplit.TRAIN: self._train_split,
            DatasetSplit.VAL: self._val_split,
            DatasetSplit.TEST: self._test_split
        }

        if dataset_ids:
            self.update_dataset(dataset_ids)

    def update_dataset(self, new_dataset: Iterable[str]) -> None:
        self._dataset = set(new_dataset)

        self._clear_splits()
        for id_ in self._dataset:
            self._add_id(id_)

    def add_instance(self, instance: str) -> DatasetSplit:
        return self._add_id(instance)

    def remove_instance(self, instance: str) -> None:
        split_set = self._split_map.get(self.get_split_for_id(instance), None)
        if split_set:
            split_set.remove(instance)

        if instance in self._dataset:
            self._dataset.remove(instance)

    def get_split(self, split: DatasetSplit) -> set[str]:
        return set(self._split_map.get(split, []))

    def get_split_ratio(self, split: DatasetSplit) -> float:
        ratio = -1

        if split == DatasetSplit.TRAIN:
            ratio = self._train_threshold
        elif split == DatasetSplit.VAL:
            ratio = self._val_threshold
        elif split == DatasetSplit.TEST:
            ratio = 1 - self._val_threshold - self._train_threshold

        return ratio

    def get_split_for_id(self, id_: str) -> Optional[DatasetSplit]:
        split = None

        items = list(self._split_map.items())
        i = 0
        while not split and i < len(items):
            current_split, ids = items[i]
            if id_ in ids:
                split = current_split
            i += 1

        return split

    def _clear_splits(self) -> None:
        """Clears the internal splits."""
        self._train_split.clear()
        self._val_split.clear()
        self._test_split.clear()

    def _add_id(self, id_: str) -> DatasetSplit:
        """Adds an ID to the appropriate split."""
        self._dataset.add(id_)
        p = self._normalized_hash(id_)

        if p < self._train_threshold:
            self._train_split.add(id_)
            result = DatasetSplit.TRAIN
        elif p < self._train_threshold + self._val_threshold:
            self._val_split.add(id_)
            result = DatasetSplit.VAL
        else:
            self._test_split.add(id_)
            result = DatasetSplit.TEST

        return result

    def _stable_hash(self, s: str) -> int:
        """Gives a stable hash for a given string."""
        key = f"{self._seed}_{s}".encode("utf-8")
        return int(hashlib.md5(key).hexdigest(), 16)

    def _normalized_hash(self, s: str) -> float:
        """Gives a normalized hash for a given string."""
        return self._stable_hash(s) / 2 ** 128
