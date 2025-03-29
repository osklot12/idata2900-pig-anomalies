import math
import threading
from abc import abstractmethod
from typing import List, TypeVar, Generic, Tuple

from src.data.data_structures.hash_buffer import HashBuffer
from src.data.dataclasses.identifiable import Identifiable
from src.data.dataclasses.streamed_annotated_frame import StreamedAnnotatedFrame
from src.data.dataset.splitters.dataset_splitter import DatasetSplitter
from src.data.dataset.dataset_split import DatasetSplit
from src.data.providers.instance_provider import InstanceProvider

I = TypeVar("I", bound=Identifiable)
O = TypeVar("O")


class VirtualDataset(Generic[I, O], InstanceProvider[O]):
    """A thread-safe, split-aware buffer system for managing annotated video frames in memory."""

    def __init__(self, splitter: DatasetSplitter, max_size: int):
        """
        Initializes a VirtualDataset instance.

        Args:
            splitter (DatasetSplitter): object responsible for managing dataset splits
            max_size (int): the maximum number of simultaneous instances stored
        """
        self._splitter = splitter

        self.lock = threading.Lock()

        # floor each max buffer size to never exceed the total max buffer size
        train_max_sources = math.floor(self._splitter.get_split_ratio(DatasetSplit.TRAIN) * max_size)
        val_max_sources = math.floor(self._splitter.get_split_ratio(DatasetSplit.VAL) * max_size)
        test_max_sources = math.floor(self._splitter.get_split_ratio(DatasetSplit.TEST) * max_size)

        # split buffers
        self._train_buffer = HashBuffer[str, O](max_size=train_max_sources)
        self._val_buffer = HashBuffer[str, O](max_size=val_max_sources)
        self._test_buffer = HashBuffer[str, O](max_size=test_max_sources)

        self._buffer_map = {
            DatasetSplit.TRAIN: self._train_buffer,
            DatasetSplit.VAL: self._val_buffer,
            DatasetSplit.TEST: self._test_buffer
        }

    def get_batch(self, split: DatasetSplit, batch_size: int) -> List[O]:
        with self.lock:
            buffer = self._buffer_map.get(split)
            if len(buffer) < batch_size:
                raise ValueError(f"Not enough available data in split {split.name} for batch size {batch_size}.")

            batch = self._get_shuffled_batch(buffer, batch_size)
            if len(batch) < batch_size:
                raise ValueError(f"Expected batch of size {batch_size} but got {len(batch)}")

            return batch

    @abstractmethod
    def _get_shuffled_batch(self, sample_buffer: HashBuffer[str, O], batch_size: int) -> List[O]:
        """
        Returns a shuffled batch of instances.

        Args:
            sample_buffer (HashBuffer[str, T]): the buffer to sample from
            batch_size (int): the batch size

        Returns:
            List[T]: the shuffled batch of instances
        """
        raise NotImplementedError

    def feed(self, food: I) -> None:
        """
        Feeds an annotated frame.

        Args:
            food (I): the data instance to feed
        """
        id_ = food.get_id()

        with self.lock:
            split_buffer = self._get_split_buffer_for_source(id_)
            evicted = split_buffer.add(*self._get_identified_instance(food))

            # remove evicted instances from the splitter
            for id_ in evicted:
                self._splitter.remove_instance(id_)

    @abstractmethod
    def _get_identified_instance(self, food: I) -> Tuple[str, O]:
        """
        Returns the instance with an identifier.

        Args:
            food (I): the fed data

        Returns:
            Tuple[str, T]: a tuple (str, T) containing the ID and instance
        """
        raise NotImplementedError

    def __len__(self) -> int:
        count = 0
        for buffer in self._buffer_map.values():
            for instance in buffer:
                count += self._frame_in_instance(instance)
        return count

    @abstractmethod
    def _frame_in_instance(self, instance: O) -> int:
        """
        Returns the number of frames in the instance.

        Args:
            instance (T): the instance to get frame count for

        Returns:
            int: the number of frames in the instance
        """
        raise NotImplementedError

    def split_size(self, split: DatasetSplit) -> int:
        """
        Returns the size of the given split.

        Args:
            split (DatasetSplit): the split to get size of

        Returns:
            int: the size of the given split
        """
        return len(self._buffer_map.get(split))

    def _get_split_buffer_for_source(self, source_id: str) -> HashBuffer:
        """Returns the split buffer for the given source id."""
        split = self._splitter.get_split_for_id(source_id)

        # updating splits if source was not recognized
        if split is None:
            split = self._splitter.add_instance(source_id)

        if split is None:
            raise RuntimeError(f"Splitter failed to assign a split for source: {source_id}")

        buffer = self._buffer_map.get(split)

        return buffer
