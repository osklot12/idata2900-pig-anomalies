from torch.utils.data import IterableDataset
from src.models.converters.xi.ultralytics_batch_converter import UltralyticsBatchConverter
from src.data.dataset.streams.prefetcher import Prefetcher
from src.data.dataclasses.annotated_frame import AnnotatedFrame  # needed for type hint
from typing import List


class EvalUltralyticsDataset(IterableDataset):
    def __init__(self, prefetcher: Prefetcher[AnnotatedFrame], batch_size: int = 8, num_batches: int = 6495):
        super().__init__()
        self._prefetcher = prefetcher
        self._batch_size = batch_size
        self._num_batches = num_batches

    def __iter__(self):
        for _ in range(self._num_batches):
            batch = [self._prefetcher.read() for _ in range(self._batch_size)]
            for sample in UltralyticsBatchConverter.convert(batch):
                yield sample

    def __len__(self):
        return self._num_batches * self._batch_size