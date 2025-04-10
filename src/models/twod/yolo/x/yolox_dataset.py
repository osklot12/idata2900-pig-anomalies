from torch.utils.data import IterableDataset

from src.data.streaming.prefetchers.batch_prefetcher import BatchPrefetcher
from src.models.converters.yolox_batch_converter import YOLOXBatchConverter


class YOLOXDataset(IterableDataset):
    """A dataset for YOLOX."""

    def __init__(self, prefetcher: BatchPrefetcher, n_batches: int):
        """
        Initializes a YOLOXDataset instance.

        Args:
            prefetcher (BatchPrefetcher): the prefetcher to fetch data with
        """
        super().__init__()
        self._prefetcher = prefetcher
        self._n_batches = n_batches

        self.class_ids = [0, 1, 2, 3]
        self.class_names = ["tail_biting", "ear_biting", "belly_nosing", "tail_down"]

    def __iter__(self):
        count = 0
        while count < len(self):
            batch = self._prefetcher.get()
            yield YOLOXBatchConverter.convert(batch)
            count += 1

    def __len__(self):
        return self._n_batches