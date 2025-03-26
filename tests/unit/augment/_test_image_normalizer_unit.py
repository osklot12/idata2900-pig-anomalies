import numpy as np
import pytest
from src.data.augment.image_normalizer import ImageNormalizer

@pytest.mark.parametrize("norm_range", [(0, 1), (-1, 1)])
def test_normalization(norm_range):
    """Test if the image normalization falls within the correct range."""
    normalizer = ImageNormalizer(norm_range=norm_range)
    sample_image = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)

    normalized_image = normalizer.process(sample_image)
    min_val, max_val = norm_range

    assert np.min(normalized_image) >= min_val, "Normalization min value incorrect"
    assert np.max(normalized_image) <= max_val, "Normalization max value incorrect"
