import math
import random
from collections import deque
from typing import List, Deque

import numpy as np

from src.data.dataset_source import DatasetSource
from src.data.frame_buffer import FrameBuffer


class VirtualDataset:
    """
    Organizes frames into train, validation, and test sets while
    mimicking a traditional dataset stored on disk.
    """

    def __init__(self, loader: "DatasetSource", max_buffer_bytes: int,
                 train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42):
        if train_ratio + val_ratio + test_ratio != 1:
            raise ValueError("Train, validation, and test ratios must sum to 1.")

        self.dataset_source = loader
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.seed = seed

        self.train_ids: List[str] = []
        self.val_ids: List[str] = []
        self.test_ids: List[str] = []

        self._shuffle_and_split()

        # floor each max buffer size to never exceed the total max buffer size
        self.train_max_bytes = math.floor(train_ratio * max_buffer_bytes)
        self.val_max_bytes = math.floor(val_ratio * max_buffer_bytes)
        self.test_max_bytes = math.floor(test_ratio * max_buffer_bytes)

        self.train_buffer = FrameBuffer(max_bytes=self.train_max_bytes)
        self.val_buffer = FrameBuffer(max_bytes=self.val_max_bytes)
        self.test_buffer = FrameBuffer(max_bytes=self.test_max_bytes)

    def feed(self, source: str, frame_index: int, frame: np.ndarray, end_of_stream: bool):
        """
        Feeds a frame into the appropriate buffer.

        :param source: The ID of the source of the frame.
        :param frame_index: The index of the frame within the source.
        :param frame: The frame data.
        :param end_of_stream: Boolean indicating whether the frame is the last in the source.
        """
        if source in self.train_ids:
            self.train_buffer.add_frame(source, frame_index, frame)
            split = "train"
        elif source in self.val_ids:
            self.val_buffer.add_frame(source, frame_index, frame)
            split = "val"
        elif source in self.test_ids:
            self.test_buffer.add_frame(source, frame_index, frame)
            split = "test"
        else:
            raise ValueError(f"Source {source} not found in dataset.")

        # Print info for debugging
        print(f"Frame {frame_index} from {source} added to {split} buffer.")

        if end_of_stream:
            print(f"End of stream for {source}.")

    def _shuffle_and_split(self):
        """Shuffles and splits the dataset into train, validation, and test sets."""
        all_files = self.dataset_source.list_files()

        random.seed(self.seed)
        random.shuffle(all_files)

        # getting train and val split end indexes, test split follows implicitly
        train_end, val_end = self._get_split_indices(all_files)

        self.train_ids = all_files[:train_end]
        self.val_ids = all_files[train_end:val_end]
        self.test_ids = all_files[val_end:]

    def _get_split_indices(self, all_files):
        """Computes the indices for the end of train and validation splits."""
        num_files = len(all_files)
        train_end = int(self.train_ratio * num_files)
        val_end = train_end + int(self.val_ratio * num_files)

        return train_end, val_end
