from typing import Tuple

from src.data.preprocessing.resizing.factories.frame_resizer_factory import FrameResizerFactory
from src.data.processing.frame_resizer import FrameResizer
from src.data.processing.frame_resizer import StaticFrameResizer


class StaticFrameResizerFactory(FrameResizerFactory):
    """A factory for creating static frame resizers."""

    def __init__(self, resize_shape: Tuple[int, int]):
        """
        Initializes a StaticFrameResizerFactory instance.

        Args:
            resize_shape (Tuple[int, int]): the shape to resize the frame to
        """
        self._resize_shape = resize_shape

    def create_frame_resizer(self) -> FrameResizer:
        return StaticFrameResizer(self._resize_shape)