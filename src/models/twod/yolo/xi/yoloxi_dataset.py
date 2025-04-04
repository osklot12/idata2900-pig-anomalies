from torch.utils.data import IterableDataset
from src.models.converters.xi.ultralytics_batch_converter import UltralyticsBatchConverter
from src.data.streaming.prefetchers.batch_prefetcher import BatchPrefetcher
import time


class UltralyticsDataset(IterableDataset):
    """An iterable dataset for Ultralytics OBB training."""

    def __init__(self, prefetcher: BatchPrefetcher):
        """
        Args:
            prefetcher (BatchPrefetcher): the prefetcher that streams AnnotatedFrames
        """
        super().__init__()
        self._prefetcher = prefetcher

    def __iter__(self):
        while True:
            batch = self._prefetcher.get()
            yield UltralyticsBatchConverter.convert(batch)  # no `from`

    def __len__(self):
        # Return a dummy large number to keep YOLO happy
        return 1_000_000