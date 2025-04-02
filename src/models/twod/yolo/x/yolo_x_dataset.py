from typing import List, Dict

from torch.utils.data import IterableDataset

from src.data.dataclasses.annotated_frame import AnnotatedFrame
from src.data.streaming.prefetchers.batch_prefetcher import BatchPrefetcher


class YOLOXDataset(IterableDataset):
    """A dataset for YOLOX."""

    def __init__(self, prefetcher: BatchPrefetcher):
        """
        Initializes a YOLOXDataset instance.

        Args:
            prefetcher (BatchPrefetcher): the prefetcher to fetch data with
        """
        super().__init__()
        self._prefetcher = prefetcher

    def __iter__(self):
        while True:
            batch = self._prefetcher.get()

    def _convert_batch(self, batch: List[AnnotatedFrame]) -> Dict[str, object]:
        images = []
        boxes = []
        labels = []

        for annotated_frame in batch:
            images.append(annotated_frame.frame)

            frame_boxes = []
            frame_labels = []

            for ann in annotated_frame.annotations:
                x1 = ann.bbox.x
                y1 = ann.bbox.y
                x2 = x1 + ann.bbox.width
                y2 = y1 + ann.bbox.height
                frame_boxes.append([x1, y1, x2, y2])