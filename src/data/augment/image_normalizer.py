import numpy as np

class ImageNormalizer:
    """
    Handles normalization of image pixel values.
    """
    def __init__(self, norm_range=(0, 1)):
        self.norm_range = norm_range
        self._validate_norm_range()

    def process(self, image: np.ndarray):
        """
        Normalizes the image to the specified range.

        :param image: Raw RGB image as a NumPy array.
        :return: Normalized image.
        """
        min_val, max_val = self.norm_range
        image = image.astype(np.float32) / 255.0  # Scale to [0,1]
        image = image * (max_val - min_val) + min_val  # Scale to [min, max]
        return image

    def _validate_norm_range(self):
        """Ensures the normalization range is valid."""
        min_val, max_val = self.norm_range
        if min_val >= max_val:
            raise ValueError(f"Invalid normalization range: {self.norm_range}. min must be < max.")
