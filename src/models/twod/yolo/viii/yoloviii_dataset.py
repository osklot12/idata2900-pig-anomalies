# src/models/twod/yolo/viii/yoloviii_dataset.py

from torch.utils.data import IterableDataset
from src.models.converters.viii.yoloviii_batch_converter import YOLOv8BatchConverter


class YOLOv8StreamingDataset(IterableDataset):
    def __init__(self, stream_factory, batch_size, max_batches, eval_mode=False):
        self._stream_factory = stream_factory
        self._batch_size = batch_size
        self._max_batches = max_batches
        self.eval_mode = eval_mode

    def __iter__(self):
        stream = self._stream_factory.create_stream()
        i = 0

        while i < len(self):
            batch = self._fetch_batch(stream)
            if len(batch) > 0:
                yield YOLOv8BatchConverter.convert(batch)
                i += 1
            elif batch is None:
                break  # true EOS

        stream.close()

    def _fetch_batch(self, stream):
        batch = []
        while len(batch) < self._batch_size:
            frame = stream.read()
            if frame is None:
                return None  # end of stream
            batch.append(frame)
        return batch

    def __len__(self):
        return self._max_batches
