from typing import Tuple

from src.data.processing.augmentation.photometric.contrast_filter import ContrastFilter
from src.data.processing.augmentation.photometric.factories.photometric_filter_factory import \
    PhotometricFilterFactory
from src.data.processing.augmentation.photometric.photometric_filter import PhotometricFilter
from src.data.structures.random_float import RandomFloat


class ContrastFilterFactory(PhotometricFilterFactory):
    """Factory for creating ContrastFilter instances."""

    def __init__(self, alpha_range: Tuple[float, float]):
        """
        Initializes a ContrastFilterFactory instance.

        Args:
            alpha_range (Tuple[float, float]): range of alpha value
        """
        self._alpha_range = alpha_range

    def create_filter(self) -> PhotometricFilter:
        return ContrastFilter(alpha=RandomFloat(min_value=self._alpha_range[0], max_value=self._alpha_range[1]))