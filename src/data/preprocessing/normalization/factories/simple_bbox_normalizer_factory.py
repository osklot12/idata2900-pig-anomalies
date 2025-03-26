from typing import Tuple

from src.data.preprocessing.normalization.factories.bbox_normalizer_factory import BBoxNormalizerFactory
from src.data.preprocessing.normalization.normalizers.bbox_normalizer import BBoxNormalizer
from src.data.preprocessing.normalization.normalizers.simple_bbox_normalizer import SimpleBBoxNormalizer


class SimpleBBoxNormalizerFactory(BBoxNormalizerFactory):
    """A factory for simple bounding box normalizers."""

    def __init__(self, new_range: Tuple[float, float]):
        """
        Initializes a SimpleBBoxNormalizerFactory instance.

        Args:
            new_range (Tuple[float, float]): new range
        """
        self._new_range = new_range

    def create_bbox_normalizer(self) -> BBoxNormalizer:
        return SimpleBBoxNormalizer(self._new_range)