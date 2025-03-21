from typing import Tuple

from src.data.preprocessing.normalization.factories.bbox_normalizer_factory import BBoxNormalizerFactory
from src.data.preprocessing.normalization.normalizers.bbox_normalization_strategy import BBoxNormalizationStrategy
from src.data.preprocessing.normalization.normalizers.simple_bbox_normalizer import SimpleBBoxNormalizer


class SimpleBBoxNormalizerFactory(BBoxNormalizerFactory):
    """A factory for simple bounding box normalizers."""

    def __init__(self, image_dimensions: Tuple[float, float], new_range: Tuple[float, float]):
        """
        Initializes a SimpleBBoxNormalizerFactory instance.

        Args:
            image_dimensions (Tuple[float, float]): original image dimensions
            new_range (Tuple[float, float]): new range
        """
        self._image_dimensions = image_dimensions
        self._new_range = new_range

    def create_bbox_normalizer(self) -> BBoxNormalizationStrategy:
        return SimpleBBoxNormalizer(self._image_dimensions, self._new_range)