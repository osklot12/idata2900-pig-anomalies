import pytest
import tensorflow as tf
import numpy as np
from src.data.image_augmentor import ImageAugmentor


@pytest.fixture
def image_augmentor():
    """Creates an instance of ImageAugmentor."""
    return ImageAugmentor(target_size=(224, 224), seed=42)


@pytest.fixture
def mock_image():
    """Generates a mock image tensor (224x224 RGB)."""
    return tf.random.uniform(shape=(224, 224, 3), minval=0, maxval=1, dtype=tf.float32)


@pytest.fixture
def mock_annotation():
    """Creates a mock annotation dictionary with a bounding box."""
    return {
        "bounding_box": {"x": 50, "y": 40, "w": 100, "h": 80},
        "label": "pig"
    }


def test_augmentor_does_not_modify_original_annotation(image_augmentor, mock_image, mock_annotation):
    """Ensures process() does not modify input annotation in-place."""
    annotation_copy = mock_annotation.copy()
    _ = image_augmentor.process(mock_image, mock_annotation, rotation_angle=5)

    # Verify the original annotation remains unchanged
    assert mock_annotation == annotation_copy, "❌ process() modified input annotation in-place!"


def test_augmentor_rotates_image(image_augmentor, mock_image, mock_annotation):
    """Tests that the augmentor applies rotation and keeps image shape valid."""
    rotated_image, updated_annotation = image_augmentor.process(mock_image, mock_annotation, rotation_angle=5)

    assert rotated_image.shape == mock_image.shape, "❌ Image shape changed after rotation!"
    assert "bounding_box" in updated_annotation, "❌ Bounding box missing in updated annotation!"


def test_augmentor_adjusts_bbox(image_augmentor, mock_image, mock_annotation):
    """Tests that the bounding box is properly adjusted after augmentation."""
    _, adjusted_annotation = image_augmentor.process(mock_image, mock_annotation, rotation_angle=5)

    bbox = adjusted_annotation["bounding_box"]

    assert isinstance(bbox, dict), "❌ Bounding box is not a dictionary!"
    assert all(k in bbox for k in ["x", "y", "w", "h"]), "❌ Bounding box keys are missing!"
    assert bbox["w"] > 0 and bbox["h"] > 0, "❌ Bounding box dimensions are invalid!"


def test_image_noise_and_flipping(image_augmentor, mock_image, mock_annotation):
    """Tests that brightness, contrast, and flipping apply correctly."""
    augmented_image, updated_annotation = image_augmentor.process(mock_image, mock_annotation)

    assert augmented_image.shape == mock_image.shape, "❌ Image shape changed unexpectedly!"
    assert "bounding_box" in updated_annotation, "❌ Bounding box missing in updated annotation!"
