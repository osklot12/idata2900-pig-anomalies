import numpy as np
import random
from typing import List

from src.data.dataclasses.annotated_frame import AnnotatedFrame
from src.data.dataclasses.annotated_bbox import AnnotatedBBox
from src.data.dataclasses.bbox import BBox
from src.data.dataclasses.annotation_class import AnnotationClass
from src.data.streaming.prefetchers.prefetcher import Prefetcher


class FakeBatchPrefetcher(Prefetcher[List[AnnotatedFrame]]):
    def __init__(self, batch_size: int, image_size=(640, 640)):
        self.batch_size = batch_size
        self.image_size = image_size

    def run(self):
        pass  # no background thread needed

    def stop(self):
        pass

    def get(self) -> List[AnnotatedFrame]:
        batch = []
        for _ in range(self.batch_size):
            h, w = self.image_size
            # Generate a fake RGB image
            frame = (np.random.rand(h, w, 3) * 255).astype(np.uint8)

            num_boxes = random.randint(0, 5)
            annotations = []
            for _ in range(num_boxes):
                x = random.uniform(0, w * 0.8)
                y = random.uniform(0, h * 0.8)
                bw = random.uniform(10, w * 0.2)
                bh = random.uniform(10, h * 0.2)
                cls = random.choice(list(AnnotationClass))

                annotations.append(
                    AnnotatedBBox(cls=cls, bbox=BBox(x, y, bw, bh))
                )

            batch.append(AnnotatedFrame(frame=frame, annotations=annotations))

        return batch

    def reset(self):
        """Override reset to avoid resetting the data for streaming dataset."""
        pass  # This prevents the reset logic from doing anything.