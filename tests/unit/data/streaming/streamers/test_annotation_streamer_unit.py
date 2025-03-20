from unittest.mock import MagicMock

import pytest

from src.data.dataclasses.frame_annotation import FrameAnnotation
from src.data.preprocessing.normalization.simple_bbox_normalizer import SimpleBBoxNormalizer
from src.data.streaming.streamers.streamer_status import StreamerStatus
from tests.utils.dummies.dummy_annotation_streamer import DummyAnnotationStreamer


@pytest.fixture
def mock_callback():
    """Fixture to provide a mock callback."""
    return MagicMock()


@pytest.mark.parametrize("n_annotations", [0, 1, 5, 10])
def test_annotation_streamer_produces_and_feeds_expected_number_of_annotations(n_annotations, mock_callback):
    """Tests that the AnnotationStreamer produces and feeds the correct number of annotations."""
    # arrange
    streamer = DummyAnnotationStreamer(n_annotations, mock_callback)

    # act
    streamer.start_streaming()
    streamer.wait_for_completion()

    # assert
    assert mock_callback.call_count == n_annotations

    for call_args in mock_callback.call_args_list:
        annotation = call_args[0][0]
        assert isinstance(annotation, FrameAnnotation)

    assert streamer.get_status() == StreamerStatus.COMPLETED


def test_annotation_streamer_should_indicate_stopping_when_stopped_early(mock_callback):
    """Tests that the AnnotationStreamer indicates that it was stopped when stopped early."""
    # arrange
    streamer = DummyAnnotationStreamer(100, mock_callback)
    streamer.start_streaming()

    # act
    streamer.stop()

    # assert
    assert streamer.get_status() == StreamerStatus.STOPPED


def test_annotation_streamer_should_normalize_bboxes(mock_callback):
    """Tests that the AnnotationStreamer normalizes bboxes when provided a normalizer."""
    # arrange
    normalizer = SimpleBBoxNormalizer((1920, 1080), (0, 1))
    streamer = DummyAnnotationStreamer(3, mock_callback, normalizer)

    # act
    streamer.start_streaming()
    streamer.wait_for_completion()

    # assert
    assert mock_callback.call_count == 3
    for call_args in mock_callback.call_args_list:
        annotation = call_args[0][0]
        for anno_bbox in annotation.annotations:
            assert anno_bbox.bbox.center_x < 1
            assert anno_bbox.bbox.center_x > 0
            assert anno_bbox.bbox.center_y < 1
            assert anno_bbox.bbox.center_y > 0