from typing import List, Dict

import numpy as np
import torch
from torch.utils.data import IterableDataset

from src.data.dataclasses.annotated_frame import AnnotatedFrame
from src.data.streaming.prefetchers.batch_prefetcher import BatchPrefetcher
from src.models.converters.bbox_to_corners import BBoxToCorners


class YOLOXDataset(IterableDataset):
    """A dataset for YOLOX."""

    def __init__(self, prefetcher: BatchPrefetcher):
        """
        Initializes a YOLOXDataset instance.

        Args:
            prefetcher (BatchPrefetcher): the prefetcher to fetch data with
        """
        super().__init__()
        self._prefetcher = prefetcher

    def __iter__(self):
        while True:
            batch = self._prefetcher.get()
