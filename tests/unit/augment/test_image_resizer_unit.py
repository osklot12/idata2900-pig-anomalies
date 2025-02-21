import pytest
import tensorflow as tf
from src.augment.image_resizer import ImageResizer

@pytest.fixture
def test_image():
    """Creates a test image tensor (random 224x224 RGB image)."""
    return tf.random.uniform((224, 224, 3), dtype=tf.float32)

@pytest.mark.parametrize("target_size", [(128, 128), (256, 256)])
def test_image_resizer(test_image, target_size):
    """Tests if images are correctly resized."""
    resizer = ImageResizer(target_size)
    resized_image = resizer.process(test_image)

    assert resized_image.shape[:2] == target_size, f"Expected {target_size}, got {resized_image.shape[:2]}"
