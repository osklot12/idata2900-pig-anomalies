from torch.utils.data import IterableDataset

from src.data.streaming.prefetchers.batch_prefetcher import BatchPrefetcher
from src.models.converters.yolox_batch_converter import YOLOXBatchConverter


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
            print("YOLOXDataset: yielded batch")
            yield YOLOXBatchConverter.convert(batch)

    def __len__(self):
        return 1000