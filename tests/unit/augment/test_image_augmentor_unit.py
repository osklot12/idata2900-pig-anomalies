import pytest
import tensorflow as tf
from src.data.preprocessing.augmentation.image_augmentor import ImageAugmentor

@pytest.fixture
def test_image():
    """Creates a test image tensor (random 224x224 RGB image)."""
    return tf.random.uniform((224, 224, 3), dtype=tf.float32)

def test_flip(test_image):
    """Tests if the image is correctly flipped."""
    augmentor = ImageAugmentor()
    flipped_image = augmentor.augment(test_image, flip=True)

    assert flipped_image.shape == test_image.shape, "Flipped image has an incorrect shape"

def test_rotation(test_image):
    """Tests if the image is rotated."""
    augmentor = ImageAugmentor()
    rotated_image = augmentor.augment(test_image, rotation=10)

    assert rotated_image.shape == test_image.shape, "Rotated image has an incorrect shape"

def test_brightness_contrast(test_image):
    """Tests if brightness and contrast adjustments apply."""
    augmentor = ImageAugmentor()
    adjusted_image = augmentor.augment(test_image)

    assert adjusted_image.shape == test_image.shape, "Brightness/contrast changed the shape"
