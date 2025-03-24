import pytest
import tensorflow as tf
from src.data.preprocessing.augmentation.combined_augmentor import AugmentationPipeline

@pytest.fixture
def test_image():
    """Creates a test image tensor (random 224x224 RGB image)."""
    return tf.random.uniform((224, 224, 3), dtype=tf.float32)

@pytest.fixture
def mock_annotation():
    """Creates a mock annotation with a bounding box."""
    return {"bounding_box": {"x": 50, "y": 50, "w": 100, "h": 100}}

@pytest.mark.parametrize("rotation", [0, 10])
@pytest.mark.parametrize("flip", [True, False])
def test_augmentation_pipeline(test_image, mock_annotation, rotation, flip):
    """Tests if the augmentation pipeline correctly applies transformations."""
    pipeline = AugmentationPipeline()
    aug_image, aug_annotation = pipeline.augment(test_image, mock_annotation)

    assert isinstance(aug_image, tf.Tensor), "Augmented image is not a tensor"
    assert aug_image.shape == test_image.shape, "Augmented image shape mismatch"
    assert "bounding_box" in aug_annotation, "Bounding box missing"
