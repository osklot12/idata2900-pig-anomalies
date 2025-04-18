from typing import TypeVar, List, Optional

from torch.utils.data import IterableDataset

from src.data.dataset.streams.closable_stream import ClosableStream
from src.data.dataset.streams.factories.stream_factory import ClosableStreamFactory
from src.models.converters.yolox_batch_converter import YOLOXBatchConverter

# data type for the data
T = TypeVar("T")


class YOLOXDataset(IterableDataset):
    """A dataset for YOLOX."""

    def __init__(self, stream_factory: ClosableStreamFactory[T], batch_size: int, n_batches: int):
        """
        Initializes a YOLOXDataset instance.

        Args:
            stream_factory (ClosableStreamFactory[T]): factory for creating dataset streams
            batch_size (int): the batch size
            n_batches (int): the number of total batches
        """
        super().__init__()
        self._stream_factory = stream_factory
        self._batch_size = batch_size
        self._n_batches = n_batches

        self.class_ids = [0, 1, 2, 3]
        self.class_names = ["tail_biting", "ear_biting", "belly_nosing", "tail_down"]

    def __iter__(self):
        print("==> __iter__ called")
        stream = self._stream_factory.create_stream()

        i = 0
        eos = False
        while i < len(self) and not eos:
            batch = self._fetch_batch(stream)
            if len(batch) > 0:
                yield YOLOXBatchConverter.convert(batch)
                i += 1

            if len(batch) < self._batch_size:
                eos = True

        stream.close()

    def _fetch_batch(self, stream: ClosableStream[T]) -> List[T]:
        """Fetches the next batch."""
        batch = []

        eos = False
        while len(batch) < self._batch_size and not eos:
            instance = stream.read()
            if instance is not None:
                batch.append(instance)
            else:
                eos = True

        return batch

    def __len__(self):
        return self._n_batches