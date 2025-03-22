import numpy as np
import pytest

from src.data.preprocessing.resizing.resizers.static_frame_resizer import StaticFrameResizer


@pytest.fixture
def sample_frame():
    """Fixture to provide a sample frame."""
    return np.random.randint(0, 256, size=(240, 320, 3), dtype=np.uint8)


def test_resize_frame_correct_shape(sample_frame):
    """Tests that the frame is resized correctly to the target shape."""
    # arrange
    target_shape = (160, 120)
    resizer = StaticFrameResizer(target_shape)

    # act
    resized = resizer.resize_frame(sample_frame)

    # assert
    assert resized.shape == (target_shape[1], target_shape[0], 3)


def test_resize_frame_raises_on_none():
    """Tests that passing None as frame_data raises ValueError."""
    # assert
    resizer = StaticFrameResizer((120, 160))

    # act & assert
    with pytest.raises(ValueError, match="frame_data cannot be None"):
        resizer.resize_frame(None)


def test_resize_frame_same_shape(sample_frame):
    """Tests that resizing a frame to the same shape still returns a valid frame."""
    # arrange
    shape = (320, 240)
    resizer = StaticFrameResizer(shape)

    # act
    resized = resizer.resize_frame(sample_frame)

    # assert
    assert resized.shape == (shape[1], shape[0], 3)


def test_resize_grayscale_frame():
    """Tests resizing a grayscale (2D) frame to a new shape."""
    # arrange
    frame = np.random.randint(0, 256, size=(100, 200), dtype=np.uint8)
    target_shape = (100, 50)
    resizer = StaticFrameResizer(target_shape)

    resized = resizer.resize_frame(frame)

    # assert
    resized.shape = (target_shape[1], target_shape[0])


def test_resize_single_channel():
    """Tests resizing a single-channel image."""
    # arrange
    frame = np.random.randint(0, 256, size=(240, 320, 1), dtype=np.uint8)
    target_shape = (80, 60)
    resizer = StaticFrameResizer(target_shape)

    # act
    resized = resizer.resize_frame(frame)

    # assert
    assert resized.shape == (60, 80, 1)