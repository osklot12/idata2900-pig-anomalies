from unittest.mock import Mock

import pytest
import numpy as np

from src.data.loading.frame_annotation_loader import FrameAnnotationLoader
from src.data.loading.feed_status import FeedStatus
from src.utils.norsvin_behavior_class import NorsvinBehaviorClass

@pytest.fixture
def mock_callback():
    """Creates a mock callback function."""
    return Mock(return_value=FeedStatus.ACCEPT)

@pytest.fixture
def loader(mock_callback):
    """Creates a FrameAnnotationLoader instance for testing."""
    return FrameAnnotationLoader(callback=mock_callback, buffer_size=3)

def test_frame_before_annotation(loader, mock_callback):
    """Tests feeding a frame before its annotation."""
    # arrange
    test_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    test_annotations = [(NorsvinBehaviorClass.TAIL_BITING, 0.1, 0.2, 0.3, 0.4)]

    # act
    feed_frame_status = loader.feed_frame("video1.mp4", 42, test_frame, False)
    after_frame_call_count = mock_callback.call_count

    feed_annotation_status = loader.feed_annotation("video1.mp4", 42, test_annotations, False)
    after_annotation_call_count = mock_callback.call_count

    # assert
    assert feed_frame_status == FeedStatus.ACCEPT
    assert after_frame_call_count == 0

    assert feed_annotation_status == FeedStatus.ACCEPT
    assert after_annotation_call_count == 1

def test_annotation_before_frame(loader, mock_callback):
    """Tests feeding an annotation before its frame."""
    # arrange
    test_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    test_annotations = [(NorsvinBehaviorClass.TAIL_BITING, 0.1, 0.2, 0.3, 0.4)]

    # act
    feed_annotation_status = loader.feed_annotation("video1.mp4", 42, test_annotations, False)
    after_annotation_call_count = mock_callback.call_count

    feed_frame_status = loader.feed_frame("video1.mp4", 42, test_frame, False)
    after_frame_call_count = mock_callback.call_count

    # assert
    assert feed_annotation_status == FeedStatus.ACCEPT
    assert after_annotation_call_count == 0

    assert feed_frame_status == FeedStatus.ACCEPT
    assert after_frame_call_count == 1

def test_unmatched_items_stay_in_buffer(loader):
    """Ensure unmatched frames/annotations stay in the buffer until a match arrives."""
    # arrange
    test_frame = np.zeros((720, 1280, 3), dtype=np.uint8)

    # act
    loader.feed_frame("video1.mp4", 42, test_frame, False)
    loader.feed_annotation("video1.mp4", 43, None, False)

    # assert
    assert loader.frame_buffer.has(42)
    assert loader.annotation_buffer.has(43)

def test_eviction_policy(loader):
    """Ensure old unmatched items are evicted when buffer size is exceeded."""
    # arrange
    test_frame = np.zeros((720, 1280, 3), dtype=np.uint8)

    # act
    for i in range(5):
        loader.feed_frame("video1.mp4", i, test_frame, False)

    # assert
    assert not loader.frame_buffer.has(0)
    assert loader.frame_buffer.has(4)