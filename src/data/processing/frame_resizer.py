from typing import Tuple

import numpy as np
import cv2

from src.data.processing.processor import Processor, I, O


class FrameResizer(Processor[np.ndarray, np.ndarray]):
    """A fixed-dimension frame resizer."""

    def __init__(self, resize_shape: Tuple[int, int]):
        """
        Initializes a StaticFrameResizer instance.

        Args:
            resize_shape (Tuple[int, int]): the shape (width, height) of the resized frame
        """
        self._resize_shape = resize_shape

    def process(self, data: np.ndarray) -> np.ndarray:
        if data is None:
            raise ValueError("frame_data cannot be None")

        resized = cv2.resize(data, self._resize_shape)

        if data.ndim == 3 and data.shape[2] == 1:
            resized = resized[:, :, np.newaxis]

        return resized