from typing import Tuple

import numpy as np
import cv2

from src.data.preprocessing.resizing.resizers.frame_resize_strategy import FrameResizeStrategy


class StaticFrameResizer(FrameResizeStrategy):
    """A frame resize strategy that resizes frames to a static shape."""

    def __init__(self, resize_shape: Tuple[int, int]):
        """
        Initializes a StaticFrameResizer instance.

        Args:
            resize_shape (Tuple[int, int]): the shape (height, width) of the resized frame
        """
        self._resize_shape = resize_shape

    def resize_frame(self, frame_data: np.ndarray) -> np.ndarray:
        if frame_data is None:
            raise ValueError("frame_data cannot be None")

        resized = cv2.resize(frame_data, self._resize_shape)

        if frame_data.ndim == 3 and frame_data.shape[2] == 1:
            resized = resized[:, :, np.newaxis]

        return resized