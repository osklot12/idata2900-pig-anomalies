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
        eos = False

        while i < len(self) and not eos:
            batch = self._fetch_batch(stream)
            if len(batch) > 0:
                yield YOLOv8BatchConverter.convert(batch)
                i += 1

            if len(batch) < self._batch_size:
                eos = True

        stream.close()

    def _fetch_batch(self, stream):
        batch = []
        eos = False
        while len(batch) < self._batch_size and not eos:
            frame = stream.read()
            if frame and frame.annotations:
                batch.append(frame)
            elif frame is None:
                eos = True
        return batch

    def __len__(self):
        return self._max_batches
