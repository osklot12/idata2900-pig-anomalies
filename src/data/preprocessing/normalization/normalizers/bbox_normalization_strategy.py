from abc import ABC, abstractmethod

from src.data.dataclasses.bounding_box import BoundingBox


class BBoxNormalizationStrategy(ABC):
    """A strategy for normalizing bounding boxes."""

    @abstractmethod
    def normalize_bounding_box(self, bounding_box: BoundingBox) -> BoundingBox:
        """
        Normalizes a bounding box.

        Args:
            bounding_box (BoundingBox): the bounding box to normalize

        Returns:
            BoundingBox: the normalized bounding box
        """
        raise NotImplementedError