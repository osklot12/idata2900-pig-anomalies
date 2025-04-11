from abc import ABC, abstractmethod

import numpy as np


class FrameResizer(ABC):
    """A strategy for resizing video frames."""

    @abstractmethod
    def resize_frame(self, frame_data: np.ndarray) -> np.ndarray:
        """
        Resizes the frame data dimensions.

        Args:
            frame_data (np.ndarray): the frame data to resize
        """
        raise NotImplementedError