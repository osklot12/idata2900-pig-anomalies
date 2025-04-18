from typing import TypeVar, Generic, List, Optional

from torch.utils.data import IterableDataset

from src.data.dataset.streams.closable_stream import ClosableStream
from src.data.dataset.streams.factories.stream_factory import ClosableStreamFactory
from src.models.converters.yolox_batch_converter import YOLOXBatchConverter

# data type for the data
T = TypeVar("T")


class YOLOXDataset(Generic[T], IterableDataset):
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
        self._stream: Optional[ClosableStream[T]] = None
        self._batch_size = batch_size
        self._n_batches = n_batches

        self.class_ids = [0, 1, 2, 3]
        self.class_names = ["tail_biting", "ear_biting", "belly_nosing", "tail_down"]

    def __iter__(self):
        if self._stream is None:
            self._stream = self._stream_factory.create_stream()

        i = 0
        while i < len(self) and self._stream is not None:
            batch = self._fetch_batch()
            if len(batch) == self._batch_size:
                yield YOLOXBatchConverter.convert(batch)
                i += 1

            else:
                self._stream.close()
                self._stream = None

    def _fetch_batch(self) -> List[T]:
        """Fetches the next batch."""
        batch = []

        instance = self._stream.read()
        while len(batch) < self._batch_size and instance is not None:
            batch.append(instance)
            instance = self._stream.read()

        return batch

    def __len__(self):
        return self._n_batches