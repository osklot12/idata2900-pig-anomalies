from typing import Tuple

from src.data.processing.augmentation.photometric.brightness_filter import BrightnessFilter
from src.data.processing.augmentation.photometric.factories.photometric_filter_factory import \
    PhotometricFilterFactory
from src.data.processing.augmentation.photometric.photometric_filter import PhotometricFilter
from src.data.structures.random_float import RandomFloat


class BrightnessFilterFactory(PhotometricFilterFactory):
    """Factory for creating BrightnessFilter instances."""

    def __init__(self, beta_range: Tuple[float, float]):
        """
        Initializes a BrightnessFilter instance.

        Args:
            beta_range (Tuple[float, float]): range for the beta value
        """
        self._beta_range = beta_range

    def create_filter(self) -> PhotometricFilter:
        return BrightnessFilter(RandomFloat(min_value=self._beta_range[0], max_value=self._beta_range[1]))