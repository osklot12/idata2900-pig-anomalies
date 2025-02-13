import pytest
import tensorflow as tf
import os
from src.data.image_processor import ImageProcessor

TEST_FRAMES_DIR = "../../data/test_frames"

@pytest.mark.parametrize("size", [(128, 128), (256, 256), (512, 512)])
def test_resize(size):
    """Test if the image resizing works correctly."""
    processor = ImageProcessor(target_size=size)
    sample_image = tf.random.uniform((512, 512, 3), dtype=tf.float32)
    processed_image = processor.process(sample_image)

    assert processed_image.shape[:2] == size, f"Expected {size}, got {processed_image.shape[:2]}"

@pytest.mark.parametrize("norm_range", [(0, 1), (-1, 1), (0, 255)])
def test_normalization(norm_range):
    """Test if the image normalization falls within the correct range."""
    processor = ImageProcessor(target_size=(224, 224), norm_range=norm_range)
    sample_image = tf.random.uniform((512, 512, 3), dtype=tf.float32)
    processed_image = processor.process(sample_image)

    min_val, max_val = norm_range
    assert tf.reduce_min(processed_image).numpy() >= min_val, "Normalization min value incorrect"
    assert tf.reduce_max(processed_image).numpy() <= max_val, "Normalization max value incorrect"

@pytest.fixture
def test_images():
    """Loads all test images as TensorFlow tensors."""
    images = []
    for filename in os.listdir(TEST_FRAMES_DIR):
        if filename.endswith(".png") or filename.endswith(".jpg"):  # Supports PNG & JPEG
            image_path = os.path.join(TEST_FRAMES_DIR, filename)
            image_bytes = tf.io.read_file(image_path)
            if filename.endswith(".png"):
                image_tensor = tf.image.decode_png(image_bytes, channels=3)
            else:
                image_tensor = tf.image.decode_jpeg(image_bytes, channels=3)
            images.append(image_tensor)
    return images

@pytest.mark.parametrize("size", [(128, 128), (256, 256)])
def test_resize_multiple_frames(test_images, size):
    """Test resizing multiple frames from the test folder."""
    processor = ImageProcessor(target_size=size)

    for idx, image in enumerate(test_images):
        processed_image = processor.process(image)
        assert processed_image.shape[:2] == size, f"Frame {idx} resizing failed. Expected {size}, got {processed_image.shape[:2]}"

@pytest.mark.parametrize("norm_range", [(0, 1), (-1, 1)])
def test_normalization_multiple_frames(test_images, norm_range):
    """Test normalization of multiple frames within expected range."""
    processor = ImageProcessor(target_size=(224, 224), norm_range=norm_range)

    for idx, image in enumerate(test_images):
        processed_image = processor.process(image)
        min_val, max_val = norm_range
        assert tf.reduce_min(processed_image).numpy() >= min_val, f"Frame {idx} min value incorrect"
        assert tf.reduce_max(processed_image).numpy() <= max_val, f"Frame {idx} max value incorrect"

def test_dynamic_configuration():
    """Test if the configure() function correctly updates the processor settings."""
    processor = ImageProcessor(target_size=(224, 224), norm_range=(0, 1))
    processor.configure(target_size=(512, 512), norm_range=(-1, 1))

    assert processor.target_size == (512, 512), "Target size did not update correctly"
    assert processor.norm_range == (-1, 1), "Normalization range did not update correctly"

def test_invalid_normalization():
    """Test if invalid normalization ranges raise an error."""
    with pytest.raises(ValueError):
        ImageProcessor(target_size=(224, 224), norm_range=(1, -1))  # min > max

def test_consistent_processing():
    """Test if processing the same image twice gives the same result."""
    processor = ImageProcessor(target_size=(224, 224), norm_range=(0, 1))
    sample_image = tf.random.uniform((512, 512, 3), dtype=tf.float32)

    processed_1 = processor.process(sample_image)
    processed_2 = processor.process(sample_image)

    assert tf.reduce_all(tf.equal(processed_1, processed_2)), "Processing is inconsistent"

def test_invalid_normalization():
    """Test if invalid normalization ranges raise an error."""
    with pytest.raises(ValueError, match="Invalid normalization range"):
        ImageProcessor(target_size=(224, 224), norm_range=(1, -1))  # min > max

@pytest.mark.parametrize("invalid_input", [None, 123, "not_a_tensor", [1, 2, 3]])
def test_invalid_input_type(invalid_input):
    """Test if non-tensor inputs raise a TypeError."""
    processor = ImageProcessor()
    with pytest.raises(TypeError, match="Expected TensorFlow tensor"):
        processor.process(invalid_input)

def test_empty_tensor():
    """Test if empty tensors raise an error."""
    processor = ImageProcessor()
    empty_tensor = tf.constant([], dtype=tf.float32)  # Empty tensor
    with pytest.raises(ValueError, match="Input tensor is empty"):
        processor.process(empty_tensor)

@pytest.mark.parametrize("invalid_shape", [
    tf.random.uniform((224, 224)),  # Missing channel dimension
    tf.random.uniform((224, 224, 224, 3)),  # Extra batch dimension
])
def test_invalid_shape(invalid_shape):
    """Test if tensors with incorrect shapes raise an error."""
    processor = ImageProcessor()
    with pytest.raises(ValueError, match="Expected 3D image tensor"):
        processor.process(invalid_shape)
