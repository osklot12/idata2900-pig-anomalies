from torch.utils.data import IterableDataset

from src.data.dataset.streams.closable_stream import ClosableStream
from src.models.converters.xi.ultralytics_batch_converter import UltralyticsBatchConverter
from src.data.dataset.streams.prefetcher import Prefetcher
from src.data.dataclasses.annotated_frame import AnnotatedFrame


class UltralyticsDataset(IterableDataset):
    def __init__(self, stream: ClosableStream, num_batches: int = 6495):
        super().__init__()
        self._stream = stream
        self._num_batches = num_batches

    def __iter__(self):
        for _ in range(self._num_batches):
            batch = self._stream.read()
            for sample in UltralyticsBatchConverter.convert(batch):
                yield sample
