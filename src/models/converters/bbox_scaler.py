from abc import ABC, abstractmethod

from src.data.dataclasses.bbox import BBox


class BBoxScaler(ABC):
    """Scales bounding box values.s"""

    @abstractmethod
    def scale(self, bbox: BBox) -> BBox:
        """
        Scales a bounding box.

        Args:
            bbox (BBox): the bounding box to scale

        Returns:
            BBox: the scaled bounding box
        """
        raise NotImplementedError