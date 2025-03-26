from unittest.mock import MagicMock

import pytest

from src.data.dataset.entities.lazy_video_annotations import LazyVideoAnnotations
from src.data.streaming.streamers.video_annotations_streamer import VideoAnnotationsStreamer
from tests.utils.dummies.dummy_video_annotations_loader import DummyVideoAnnotationsLoader


@pytest.fixture
def mock_callback():
    """Fixture to provide a mock callback."""
    return MagicMock()


@pytest.fixture
def dummy_annotations():
    """Fixture to provide a test annotation object."""
    return LazyVideoAnnotations("test-annotations", DummyVideoAnnotationsLoader())


def test_annotations_streamers_streams_all_annotations(dummy_annotations, mock_callback):
    """Tests that VideoAnnotationsStreamer streams all annotations as expected."""
    # arrange
    streamer = VideoAnnotationsStreamer(dummy_annotations, mock_callback)
    expected_annotations = dummy_annotations.get_data()

    # act
    streamer.start_streaming()
    streamer.wait_for_completion()

    # assert
    assert mock_callback.call_count == len(expected_annotations)

    actual_annotations = [call_args[0][0] for call_args in mock_callback.call_args_list]
    assert actual_annotations == expected_annotations
