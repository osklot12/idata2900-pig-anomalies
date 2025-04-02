from abc import ABC, abstractmethod
from typing import Tuple

from src.data.dataclasses.bbox import BBox


class BBoxConverter(ABC):
    """Interface for bounding box converters."""

    @abstractmethod
    def convert(self, bbox: BBox) -> Tuple[float, float, float, float]:
        """
        Converts a bounding box to a different format.

        Args:
            bbox (BBox): the bounding box to convert

        Returns:
            Tuple[float, float, float, float]: the converted bounding box
        """
        raise NotImplementedError