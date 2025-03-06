from unittest.mock import Mock

import pytest

from src.data.decoders.darwin_decoder import DarwinDecoder
from src.data.loading.annotation_loader import AnnotationLoader
from tests.utils.dummies.dummy_gcp_data_loader import DummyGCPDataLoader


@pytest.fixture
def dummy_data_loader():
    """Fixture to provide a dummy GCPDataLoader."""
    dummy_loader = DummyGCPDataLoader(bucket_name="dummy-bucket", credentials_path="dummy-credentials.json")
    return dummy_loader


@pytest.fixture
def mock_callback():
    """Fixture to provide a mock callback."""
    return Mock()


def test_callback_called_correctly(dummy_data_loader, mock_callback):
    """Tests that the callback function is called correctly."""
    # arrange
    annotation_loader = AnnotationLoader(
        data_loader=dummy_data_loader,
        decoder_cls=DarwinDecoder,
        callback=mock_callback
    )

    # act
    annotation_loader.load_annotations("test_annotations.json")
    annotation_loader.wait_for_completion()

    # assert
    total_frames = 171
    expected_calls = total_frames + 1

    assert mock_callback.call_count == expected_calls, \
        f"Expected {expected_calls} calls, but got {mock_callback.call_count}"

    last_call = mock_callback.call_args_list[-1]
    _, last_frame_index, last_annotations, is_complete = last_call[0]

    assert last_annotations is None, "Final callback did not receive None!"
    assert is_complete is True, "Final callback did not indicate completion!"


def test_annotations_correctly_parsed(dummy_data_loader, mock_callback):
    """Tests that annotations are correctly extracted and passed to the callback."""
    # arrange
    annotation_loader = AnnotationLoader(
        data_loader=dummy_data_loader,
        decoder_cls=DarwinDecoder,
        callback=mock_callback
    )

    # act
    annotation_loader.load_annotations("test_annotations.json")
    annotation_loader.wait_for_completion()

    # assert
    expected_annotations = {}
    for behavior, frames in dummy_data_loader.ANNOTATIONS:
        for frame, x, y, w, h in frames

    # assert
    total_frames = 171
    for frame_index in range(total_frames):
        call = mock_callback.call_args_list[frame_index]
        _, received_frame, received_annotations, is_complete = call[0]

        # if frame carries termination signal, it should not have an index or annotations
        if is_complete:
            assert received_frame is None, "Final callback should not have a frame index!"
            assert received_annotations is None, "Final callback should not have annotations!"
            continue

        # if frame has annotation, it should match the expected annotation
        if frame_index in expected_annotations:
            assert received_annotations == expected_annotations[frame_index], \
                f"Annotations at frame {frame_index} did not match expected annotations!"

        # if frame has no annotations, it should receive an empty list
        else:
            assert received_annotations == [], \
                f"Frame {frame_index} should have no annotations but got {received_annotations}"
