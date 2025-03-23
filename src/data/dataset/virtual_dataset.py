import math
import random
import threading
from typing import List, Tuple, Optional

import numpy as np

from src.data.data_structures.hash_buffer import HashBuffer
from src.data.dataclasses.annotated_frame import AnnotatedFrame
from src.data.dataclasses.streamed_annotated_frame import StreamedAnnotatedFrame
from src.data.dataset_split import DatasetSplit
from src.utils.norsvin_behavior_class import NorsvinBehaviorClass


class VirtualDataset:
    """
    Organizes streamed frames with annotations into train, validation, and test sets while
    mimicking a traditional dataset stored on disk.
    """

    def __init__(self, dataset_ids: List[str], max_sources: int, max_frames_per_source: int,
                 train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42):
        if train_ratio + val_ratio + test_ratio != 1:
            raise ValueError("Train, validation, and test ratios must sum to 1.")

        self.dataset_ids = dataset_ids
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.seed = seed

        self.lock = threading.Lock()

        self.train_ids: List[str] = []
        self.val_ids: List[str] = []
        self.test_ids: List[str] = []

        self._shuffle_and_split()

        # floor each max buffer size to never exceed the total max buffer size
        self.train_max_sources = math.floor(train_ratio * max_sources)
        self.val_max_sources = math.floor(val_ratio * max_sources)
        self.test_max_sources = math.floor(test_ratio * max_sources)

        # train split buffer
        self.train_buffer = HashBuffer[HashBuffer[AnnotatedFrame]](max_size=self.train_max_sources)

        # val split buffer
        self.val_buffer = HashBuffer[HashBuffer[AnnotatedFrame]](max_size=self.val_max_sources)

        # test split buffer
        self.test_buffer = HashBuffer[HashBuffer[AnnotatedFrame]](max_size=self.test_max_sources)

        self.max_frames_per_source = max_frames_per_source

    def get_shuffled_batch(self, split: DatasetSplit, batch_size: int) -> List[AnnotatedFrame]:
        """
        Samples a randomized batch of frame-annotation pairs from the specified dataset split.

        Args:
            split (DatasetSplit): the dataset split to sample from
            batch_size (int): the number of frames to sample

        Returns:
            List[AnnotatedFrame]: the randomized batch of frame-annotation pairs
        """
        with self.lock:
            if self.get_frame_count(split) < batch_size:
                raise ValueError(f"Not enough available data in split {split.name} for batch size {batch_size}.")

            buffer = self._get_buffer_for_split(split)

            frame_references = [
                source_buffer.at(frame_index)
                for source_key in buffer.keys()
                for source_buffer in [buffer.at(source_key)]
                if source_buffer and source_buffer.size() > 0
                for frame_index in source_buffer.keys()
            ]

            # ensure "non-deterministic" behavior
            random.seed(None)

            return random.sample(frame_references, batch_size)

    def get_frame_count(self, split: DatasetSplit) -> int:
        """
        Returns the total number of stored frames in the specified dataset split.

        :param split: The dataset split (TRAIN, VAL, TEST).
        :return: The total number of frames stored in the split.
        """
        buffer = self._get_buffer_for_split(split)
        total_frames = 0

        for source_key in buffer.keys():
            source_buffer = buffer.at(source_key)
            if source_buffer:
                total_frames += source_buffer.size()

        return total_frames

    def feed(self, anno_frame: StreamedAnnotatedFrame) -> None:
        """
        Feeds a frame into the appropriate buffer.

        Args:
            anno_frame (StreamedAnnotatedFrame): the data instance to feed
        """
        with self.lock:
            # get correct split
            split_buffer = self._get_buffer_for_source(anno_frame.source)

            if split_buffer and anno_frame.index < 0:
                print(f"[VirtualDataset] End of stream signal received for source {anno_frame.source}.")

            elif split_buffer:
                # allocates buffer for source if new
                source_buffer = self._get_source_buffer(anno_frame.source, split_buffer)

                # add instance to source buffer
                source_buffer.add(anno_frame.index, AnnotatedFrame(anno_frame.frame, anno_frame.annotations))

                print(f"[VirtualDataset] Instance {anno_frame.index} from source {anno_frame.source} received and stored")
                print(f"[VirtualDataset] Train split: {self.get_frame_count(DatasetSplit.TRAIN)}")
                print(f"[VirtualDataset] Val split: {self.get_frame_count(DatasetSplit.VAL)}")
                print(f"[VirtualDataset] Test split: {self.get_frame_count(DatasetSplit.TEST)}")

            else:
                print(f"[VirtualDataset] Source {anno_frame.source} not recognized.")

    def _get_source_buffer(self, source_key: str, split_buffer: HashBuffer) -> HashBuffer:
        """
        Returns the source buffer associated with the specified source key.
        Creates a new source buffer if it does not exist.

        Args:
            source_key (str): the source key
            split_buffer (HashBuffer): the source buffer to search in
        """
        if not split_buffer.has(source_key):
            new_buffer = HashBuffer[AnnotatedFrame](max_size=self.max_frames_per_source)
            split_buffer.add(source_key, new_buffer)

        return split_buffer.at(source_key)

    def _shuffle_and_split(self):
        """Shuffles and splits the dataset into train, validation, and test sets."""
        random.seed(self.seed)
        random.shuffle(self.dataset_ids)

        # getting train and val split end indexes, test split follows implicitly
        train_end, val_end = self._get_split_indices(self.dataset_ids)

        self.train_ids = self.dataset_ids[:train_end]
        self.val_ids = self.dataset_ids[train_end:val_end]
        self.test_ids = self.dataset_ids[val_end:]

    def _get_split_indices(self, all_files):
        """Computes the indices for the end of train and validation splits."""
        num_files = len(all_files)
        train_end = int(self.train_ratio * num_files)
        val_end = train_end + int(self.val_ratio * num_files)

        return train_end, val_end

    def _get_buffer_for_source(self, source):
        """
        Returns the buffer (split) the source belongs to.
        Returns None if source is not recognized.
        """
        if source in self.train_ids:
            buffer = self.train_buffer
        elif source in self.val_ids:
            buffer = self.val_buffer
        elif source in self.test_ids:
            buffer = self.test_buffer
        else:
            buffer = None

        return buffer

    def _get_buffer_for_split(self, split: DatasetSplit):
        """Returns the buffer (split) for the given DatasetSplit."""
        if split == DatasetSplit.TRAIN:
            buffer = self.train_buffer
        elif split == DatasetSplit.VAL:
            buffer = self.val_buffer
        elif split == DatasetSplit.TEST:
            buffer = self.test_buffer
        else:
            raise ValueError(f"Split {split} not found in dataset.")

        return buffer

    def get_split_for_source(self, source: str):
        """
        Returns the DatasetSplit for the given source.

        Args:
            source: The ID of the source data.

        Returns:
            The DatasetSplit for the given source, None if not recognized.
        """
        if source in self.train_ids:
            split = DatasetSplit.TRAIN
        elif source in self.val_ids:
            split = DatasetSplit.VAL
        elif source in self.test_ids:
            split = DatasetSplit.TEST
        else:
            split = None

        return split