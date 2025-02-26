import numpy as np
import pytest
from src.data.augment.image_augmentor import ImageAugmentor


@pytest.mark.parametrize("flip", [True, False])
@pytest.mark.parametrize("rotation", [-10, 0, 10])
def test_image_augmentor(flip, rotation):
    """Test flipping and rotation operations on an image."""

    augmentor = ImageAugmentor()
    sample_image = np.ones((224, 224, 3), dtype=np.uint8) * 255  # White image

    aug_image = augmentor.augment(sample_image, rotation=rotation, flip=flip)

    assert aug_image is not None, "Augment returned None!"
    assert aug_image.shape == sample_image.shape, "Augmentation altered image dimensions!"


def test_image_augmentor_brightness_contrast():
    """Test brightness and contrast augmentation."""

    augmentor = ImageAugmentor()
    sample_image = np.ones((224, 224, 3), dtype=np.uint8) * 127  # Mid-gray

    aug_image = augmentor.augment(sample_image)

    assert aug_image is not None, "Brightness/contrast augmentation returned None!"
    assert aug_image.shape == sample_image.shape, "Brightness/contrast augmentation altered image dimensions!"
    assert not np.array_equal(sample_image, aug_image), "Brightness/contrast did not change!"
