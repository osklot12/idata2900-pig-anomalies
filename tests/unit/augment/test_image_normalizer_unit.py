import pytest
import tensorflow as tf
from src.augment.image_normalizer import ImageNormalizer

@pytest.fixture
def test_image():
    """Creates a test image tensor (random 224x224 RGB image)."""
    return tf.random.uniform((224, 224, 3), dtype=tf.float32)

@pytest.mark.parametrize("norm_range", [(0, 1), (-1, 1)])
def test_image_normalizer(test_image, norm_range):
    """Tests if image normalization works within the expected range."""
    normalizer = ImageNormalizer(norm_range)
    normalized_image = normalizer.process(test_image)

    assert tf.reduce_min(normalized_image).numpy() >= norm_range[0], "Min normalization value incorrect"
    assert tf.reduce_max(normalized_image).numpy() <= norm_range[1], "Max normalization value incorrect"
