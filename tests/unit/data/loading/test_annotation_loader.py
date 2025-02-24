from unittest.mock import Mock

import pytest

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
    annotation_loader = AnnotationLoader(data_loader=dummy_data_loader, callback=mock_callback)

    # act
    annotation_loader.load_annotations("test_annotations.json")
    annotation_loader.wait_for_completion()

    # assert
    expected_frames = {56, 110, 128, 162}
    expected_calls = len(expected_frames) * 2 + 1

    assert mock_callback.call_count == expected_calls, \
        f"Expected {expected_calls} calls, got {mock_callback.call_count}"

    last_call = mock_callback.call_args_list[-1]
    _, last_frame_index, last_annotation, is_complete = last_call[0]
    assert last_annotation is None, "Final callback did not receive None!"
    assert is_complete is True, "Final callback did not indicate completion!"


def test_annotations_correctly_parsed(dummy_data_loader, mock_callback):
    """Tests that annotations are correctly extracted and passed to the callback."""
    # arrange
    annotation_loader = AnnotationLoader(data_loader=dummy_data_loader, callback=mock_callback)

    # act
    annotation_loader.load_annotations("test_annotations.json")
    annotation_loader.wait_for_completion()

    # assert
    expected_annotations = {
        56: [
            ("g2b_bellynosing", 1925.0824, 1178.3059, 108.3765, 110.6824),
            ("g2b_tailbiting", 1925.0824, 1178.3059, 108.3765, 110.6824),
        ],
        110: [
            ("g2b_bellynosing", 1925.0824, 1178.3059, 108.3765, 110.6824),
            ("g2b_tailbiting", 1925.0824, 1178.3059, 108.3765, 110.6824),
        ],
        128: [
            ("g2b_bellynosing", 1920.4706, 1194.4471, 126.8235, 112.9882),
            ("g2b_tailbiting", 1920.4706, 1194.4471, 126.8235, 112.9882),
        ],
        162: [
            ("g2b_bellynosing", 1920.4706, 1194.4471, 126.8235, 112.9882),
            ("g2b_tailbiting", 1920.4706, 1194.4471, 126.8235, 112.9882),
        ],
    }

    # assert
    for call in mock_callback.call_args_list:
        _, frame_index, annotation, is_complete = call[0]

        if is_complete:
            assert frame_index is None, "Final callback should not have a frame index!"
            assert annotation is None, "Final callback should not have an annotation!"
            continue

        assert annotation in expected_annotations[frame_index], f"Unexpected annotation at frame {frame_index}"