from torch.utils.data import IterableDataset
from src.models.converters.xi.ultralytics_batch_converter import UltralyticsBatchConverter
from src.data.dataset.streams.prefetcher import Prefetcher
from src.data.dataclasses.annotated_frame import AnnotatedFrame  # needed for type hint
from typing import List


class EvalUltralyticsDataset(IterableDataset):
    def __init__(self, prefetcher: Prefetcher[AnnotatedFrame], num_batches: int = 6495, batch_size: int = 8):
        super().__init__()
        self._prefetcher = prefetcher
        self._num_batches = num_batches
        self._batch_size = batch_size

    def __iter__(self):
        for _ in range(self._num_batches):
            frames: List[AnnotatedFrame] = []
            for _ in range(self._batch_size):
                frame = self._prefetcher.read()
                frames.append(frame)

            for sample in UltralyticsBatchConverter.convert(frames):
                yield sample

    def __len__(self):
        return self._num_batches