from typing import Tuple

import numpy as np

from src.data.processing.frame_resizer import FrameResizer


class DummyFrameResizeStrategy(FrameResizer):
    """A dummy frame resize strategy for testing purposes."""

    def __init__(self, resize_shape: Tuple[int, int]):
        """
        Initializes a DummyFrameResizeStrategy instance.

        Args:
            resize_shape (Tuple[int, int]): the shape to resize the frame to
        """
        self.resize_shape = resize_shape

    def resize_frame(self, frame_data: np.ndarray) -> np.ndarray:
        height, width = self.resize_shape
        return np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)