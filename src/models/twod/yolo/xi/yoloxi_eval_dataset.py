from torch.utils.data import IterableDataset
from src.models.converters.xi.ultralytics_batch_converter import UltralyticsBatchConverter
from src.data.streaming.prefetchers.batch_prefetcher import BatchPrefetcher


class EvalUltralyticsDataset(IterableDataset):
    """A finite iterable dataset for Ultralytics OBB evaluation."""

    def __init__(self, prefetcher: BatchPrefetcher, num_batches: int = 430):
        """
        Args:
            prefetcher (BatchPrefetcher): the prefetcher that streams AnnotatedFrames
            num_batches (int): number of batches to yield before stopping
        """
        super().__init__()
        self._prefetcher = prefetcher
        self._num_batches = num_batches

    def __iter__(self):
        for _ in range(self._num_batches):
            batch = self._prefetcher.get()
            for sample in UltralyticsBatchConverter.convert(batch):
                yield sample

    def __len__(self):
        # Technically this is not meaningful for IterableDataset,
        # but it's good practice to return the intended size.
        return self._num_batches