from abc import ABC, abstractmethod

from src.data.processing.frame_resizer import FrameResizer


class FrameResizerFactory(ABC):
    """An interface for frame resizer factories."""

    @abstractmethod
    def create_frame_resizer(self) -> FrameResizer:
        """
        Creates a FrameResizeStrategy instance.

        Returns:
            FrameResizer: a FrameResizeStrategy instance
        """
        raise NotImplementedError