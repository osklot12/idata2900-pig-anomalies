from typing import TypeVar

import torch
from torch.utils.data import IterableDataset

from src.data.dataset.streams.providers.stream_provider import StreamProvider

# data type for the data
T = TypeVar("T")


class StreamingDataset(IterableDataset):
    """An IterableDataset wrapper for streaming datasets."""

    def __init__(self, stream_provider: StreamProvider[T], n_batches: int):
        """
        Initializes a StreamingDataset instance.

        Args:
            stream_provider (StreamProvider[T]): provider of streams
            n_batches (int): the number of total batches
        """
        super().__init__()
        self._stream_provider = stream_provider
        self._n_batches = n_batches

        self.class_ids = [0, 1, 2, 3]
        self.class_names = ["tail_biting", "ear_biting", "belly_nosing", "tail_down"]

    def __iter__(self):
        stream = self._stream_provider.get_stream()

        i = 0
        instance = stream.read()
        while i < len(self) and instance is not None:
            image = instance.frame
            image_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

            _, h, w = image_tensor.shape

            boxes = []
            labels = []
            for ann in instance.annotations:
                x1 = ann.bbox.x * w
                y1 = ann.bbox.y * h
                x2 = x1 + ann.bbox.width * w
                y2 = y1 + ann.bbox.height * h
                boxes.append([x1, y1, x2, y2])
                labels.append(ann.cls.value)

            target = {
                "boxes": torch.tensor(boxes, dtype=torch.float32),
                "labels": torch.tensor(labels, dtype=torch.int64)
            }

            yield image_tensor, target
            instance = stream.read()
            i += 1

    def __len__(self):
        return self._n_batches