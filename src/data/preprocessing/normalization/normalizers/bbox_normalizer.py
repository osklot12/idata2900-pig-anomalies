from abc import ABC, abstractmethod

from src.data.dataclasses.bbox import BBox


class BBoxNormalizer(ABC):
    """A strategy for normalizing bounding boxes."""

    @abstractmethod
    def normalize_bounding_box(self, bounding_box: BBox) -> BBox:
        """
        Normalizes a bounding box.

        Args:
            bounding_box (BBox): the bounding box to normalize

        Returns:
            BBox: the normalized bounding box
        """
        raise NotImplementedError