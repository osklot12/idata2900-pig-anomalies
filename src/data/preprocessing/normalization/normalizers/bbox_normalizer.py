from abc import ABC, abstractmethod
from typing import Tuple

from src.data.dataclasses.bbox import BBox


class BBoxNormalizer(ABC):
    """A strategy for normalizing bounding boxes."""

    @abstractmethod
    def normalize_bounding_box(self, bounding_box: BBox, resolution: Tuple[int, int]) -> BBox:
        """
        Normalizes a bounding box.

        Args:
            bounding_box (BBox): the bounding box to normalize
            resolution (Tuple[int, int]): the resolution (width, height) of the associated frame

        Returns:
            BBox: the normalized bounding box
        """
        raise NotImplementedError