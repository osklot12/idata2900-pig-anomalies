from abc import ABC, abstractmethod

from src.data.preprocessing.resizing.frame_resize_strategy import FrameResizeStrategy


class FrameResizerFactory(ABC):
    """An interface for frame resizer factories."""

    @abstractmethod
    def create_frame_resizer(self) -> FrameResizeStrategy:
        """
        Creates a FrameResizeStrategy instance.

        Returns:
            FrameResizeStrategy: a FrameResizeStrategy instance
        """
        raise NotImplementedError