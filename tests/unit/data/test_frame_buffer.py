import threading

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

### thread safety tests

def add_frames_to_buffer(buffer: FrameBuffer, n_frames: int, frame_size: tuple):
    """Adds multiple frames to the buffer."""
    for i in range(n_frames):
        frame = np.random.randint(0, 255, size=frame_size, dtype=np.uint8)
        buffer.add_frame("video1.mp4", i, frame)

def test_frame_buffer_thread_safety(small_frame):
    """Tests that FrameBuffer handles multiple threads safely."""
    # arrange
    buffer_size = 10_000_000
    buffer = FrameBuffer(max_bytes=buffer_size)

    n_threads = 5
    n_frames_per_thread = 10
    frame_size = small_frame.shape

    # act
    threads = []
    for _ in range(n_threads):
        t = threading.Thread(target=add_frames_to_buffer, args=(buffer, n_frames_per_thread, frame_size))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    # assert
    assert len(buffer) > 0, "Buffer should have frames after concurrent writes."
    assert buffer.current_mem_usage <= buffer_size, f"Buffer memory exceeded: {buffer.current_mem_usage} > {buffer_size}"

