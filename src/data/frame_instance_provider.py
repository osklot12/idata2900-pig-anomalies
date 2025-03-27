from abc import ABC, abstractmethod
from typing import List

from src.data.dataclasses.annotated_frame import AnnotatedFrame
from src.data.dataset_split import DatasetSplit


class FrameInstanceProvider(ABC):
    """An interface for providers of annotated frame instances."""

    @abstractmethod
    def get_batch(self, split: DatasetSplit, batch_size: int) -> List[AnnotatedFrame]:
        """
        Returns a batch of annotated frame instances.

        Args:
            split (DatasetSplit): the split to sample from
            batch_size (int): the batch size

        Returns:
            List[AnnotatedFrame]: the batch of annotated frame instances
        """
        raise NotImplementedError

    @abstractmethod
    def get_instance(self, split: DatasetSplit) -> AnnotatedFrame:
        """
        Returns an annotated frame instance.

        Args:
            split (DatasetSplit): the split to sample from

        Returns:
            AnnotatedFrame: the annotated frame instance
        """
        raise NotImplementedError