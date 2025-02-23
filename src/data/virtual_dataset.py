import random
from collections import deque
from typing import List, Deque

import numpy as np

from src.data.loading.frame_loader_interface import FrameLoaderInterface


class VirtualDataset:
    """
    Organizes frames into train, validation, and test sets while
    mimicking a traditional dataset stored on disk.
    """

    def __init__(self, loader: "FrameLoaderInterface", max_buffer_bytes: int,
                 train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42):
        if train_ratio + val_ratio + test_ratio != 1:
            raise ValueError("Train, validation, and test ratios must sum to 1.")

        self.loader = loader
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.seed = seed

        self.train_ids: List[str] = []
        self.val_ids: List[str] = []
        self.test_ids: List[str] = []

        self._shuffle_and_split()

        self.max_buffer_bytes = max_buffer_bytes


    def feed(self, source: str, frame_index: int, frame: np.ndarray, end_of_stream: bool):
        """"""
        

    def _shuffle_and_split(self):
        """Shuffles and splits the dataset into train, validation, and test sets."""
        all_files = self.loader.get_data_source().list_files()

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