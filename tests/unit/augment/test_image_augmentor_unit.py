import numpy as np
import pytest
from src.data.augment.image_augmentor import ImageAugmentor

def test_augmentation():
    """Test flipping and rotation operations on an image."""
    augmentor = ImageAugmentor()
    sample_image = np.ones((224, 224, 3), dtype=np.uint8) * 255

    # Test flipping
    flipped_image = augmentor.augment(sample_image, flip=True)
    assert flipped_image.shape == sample_image.shape, "Flipping changed image dimensions!"

    # Test rotation (small angle, check shape consistency)
    rotated_image = augmentor.augment(sample_image, rotation=15)
    assert rotated_image.shape == sample_image.shape, "Rotation affected image dimensions!"
