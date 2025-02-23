import pytest
import numpy as np
from src.data.frame_buffer import FrameBuffer

@pytest.fixture
def small_frame():
    """Returns a small dummy frame (50KB)."""
    return np.random.randint(0, 255, size=(128, 128, 3), dtype=np.uint8)

@pytest.fixture
def large_frame():
    """Returns a large dummy frame (5MB)."""
    return np.random.randint(0, 255, (1024, 1024, 3), dtype=np.uint8)

def test_frame_buffer_addition(small_frame):
    """Tests that frames are added correctly."""
    # arrange
    buffer = FrameBuffer(max_bytes=500_000)

    # act
    buffer.add_frame("video1.mp4", 0, small_frame)

    # assert
    assert len(buffer) == 1
    assert buffer.current_mem_usage == small_frame.nbytes

def test_frame_buffer_eviction(small_frame):
    """Tests that oldest frames are evicted when the buffer exceeds the limit."""
    # arrange
    buffer = FrameBuffer(max_bytes=100_000)

    # act
    buffer.add_frame("video1.mp4", 0, small_frame)
    buffer.add_frame("video1.mp4", 1, small_frame)
    buffer.add_frame("video1.mp4", 2, small_frame)

    # assert
    assert len(buffer) == 2
    assert buffer.get_frames()[0][1] == 1

def test_frame_buffer_large_frame_drop(large_frame):
    """Tests that an oversize frame is not added to the buffer."""
    # arrange
    buffer = FrameBuffer(max_bytes=int(1e6))

    # act
    buffer.add_frame("video1.mp4", 0, large_frame)

    # assert
    assert len(buffer) == 0
    assert buffer.current_mem_usage == 0