from typing import Tuple

from src.data.processing.augmentation.photometric.factories.photometric_filter_factory import \
    PhotometricFilterFactory
from src.data.processing.augmentation.photometric.gaussian_noise_filter import GaussianNoiseFilter
from src.data.processing.augmentation.photometric.photometric_filter import PhotometricFilter
from src.data.structures.random_float import RandomFloat


class GaussianNoiseFilterFactory(PhotometricFilterFactory):
    """Factory for creating GaussianNoiseFilter instances."""

    def __init__(self, std_range: Tuple[float, float]):
        """
        Initializes a GaussianNoiseFilterFactory instance.

        Args:
            std_range (Tuple[float, float]): range for the standard deviation
        """
        self._std_range = std_range

    def create_filter(self) -> PhotometricFilter:
        return GaussianNoiseFilter(std=RandomFloat(min_value=self._std_range[0], max_value=self._std_range[1]))