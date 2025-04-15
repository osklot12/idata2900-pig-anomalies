from typing import Tuple

from src.data.preprocessing.augmentation.augmentors.photometric.color_jitter_filter import ColorJitterFilter
from src.data.preprocessing.augmentation.augmentors.photometric.factories.photometric_filter_factory import \
    PhotometricFilterFactory
from src.data.preprocessing.augmentation.augmentors.photometric.photometric_filter import PhotometricFilter
from src.data.structures.random_float import RandomFloat


class ColorJitterFilterFactory(PhotometricFilterFactory):
    """Factory for creating ColorJitterFilter instances."""

    def __init__(self, saturation_range: Tuple[float, float], hue_range: Tuple[float, float]):
        """
        Initializes a ColorJitterFilter instance.

        Args:
            saturation_range (Tuple[float, float]): range for saturation scale
            hue_range (Tuple[float, float]): range for hue scale
        """
        self._saturation_range = saturation_range
        self._hue_range = hue_range

    def create_filter(self) -> PhotometricFilter:
        saturation_float = RandomFloat(min_value=self._saturation_range[0], max_value=self._saturation_range[1])
        hue_float = RandomFloat(min_value=self._hue_range[0], max_value=self._hue_range[1])
        
        return ColorJitterFilter(saturation_scale=saturation_float, hue_shift=hue_float)
