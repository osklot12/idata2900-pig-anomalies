import pytest
import tensorflow as tf
from src.augment.json_augmentor import JsonAugmentor

@pytest.fixture
def test_image():
    """Creates a dummy TensorFlow image."""
    return tf.zeros((224, 224, 3), dtype=tf.float32)

@pytest.fixture
def mock_annotation():
    """Creates a mock annotation with a bounding box."""
    return {"bounding_box": {"x": 50, "y": 50, "w": 100, "h": 100}}

@pytest.mark.parametrize("rotation", [0, 15, -15])
@pytest.mark.parametrize("flip", [True, False])
def test_json_augmentor(test_image, mock_annotation, rotation, flip):
    """Tests if bounding box transformation applies correctly."""
    json_augmentor = JsonAugmentor()
    augmented_annotation = json_augmentor.augment(test_image, mock_annotation, rotation=rotation, flip=flip)

    assert "bounding_box" in augmented_annotation, "Bounding box missing"
    assert isinstance(augmented_annotation["bounding_box"], dict), "Bounding box format incorrect"
