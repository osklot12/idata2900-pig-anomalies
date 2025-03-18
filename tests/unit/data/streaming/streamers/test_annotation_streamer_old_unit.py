from unittest.mock import Mock

import pytest

from src.data.decoders.darwin_decoder import DarwinDecoder
from src.data.streaming.streamers import AnnotationStreamer
from src.data.preprocessing.normalization.simple_bbox_normalizer import SimpleBBoxNormalizer
from src.utils.norsvin_annotation_parser import NorsvinAnnotationParser
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
    normalizer = SimpleBBoxNormalizer((1920, 1080), (0, 1), NorsvinAnnotationParser)
    annotation_loader = AnnotationStreamer(
        data_loader=dummy_data_loader,
        annotation_blob_name="test_annotations.json",
        decoder_cls=DarwinDecoder,
        normalizer=normalizer,
        callback=mock_callback
    )

    # Act
    annotation_loader.start_streaming()
    annotation_loader.wait_for_completion()

    # Assert
    total_frames = dummy_data_loader.frame_count  # Use the stored frame count
    expected_calls = total_frames + 1  # +1 for the termination signal

    assert mock_callback.call_count == expected_calls, \
        f"Expected {expected_calls} calls, but got {mock_callback.call_count}"

    last_call = mock_callback.call_args_list[-1]
    _, last_frame_index, last_annotations, is_complete = last_call[0]

    assert last_annotations is None, "Final callback did not receive None!"
    assert is_complete is True, "Final callback did not indicate completion!"


def test_annotations_correctly_parsed(dummy_data_loader, mock_callback):
    """Tests that annotations are correctly extracted and passed to the callback."""
    # arrange
    normalizer = SimpleBBoxNormalizer((1920, 1080), (0, 1), NorsvinAnnotationParser)
    annotation_loader = AnnotationStreamer(
        data_loader=dummy_data_loader,
        annotation_blob_name="test_annotations.json",
        decoder_cls=DarwinDecoder,
        normalizer=normalizer,
        callback=mock_callback
    )

    # act
    annotation_loader.start_streaming()
    annotation_loader.wait_for_completion()

    # assert
    total_frames = dummy_data_loader.frame_count
    expected_annotations = dummy_data_loader.get_annotations()

    normalized_expected_annotations = _normalize_annotation_dict(dummy_data_loader)

    for frame_index in range(total_frames):
        call = mock_callback.call_args_list[frame_index]
        _, received_frame, received_annotations, is_complete = call[0]

        # If the final callback, ensure it's signaling completion
        if is_complete:
            assert received_frame is None, "Final callback should not have a frame index!"
            assert received_annotations is None, "Final callback should not have annotations!"
            continue

        # If the frame is expected to have annotations, check if they match
        if frame_index in normalized_expected_annotations:
            assert received_annotations == normalized_expected_annotations[frame_index], \
                f"Annotations at frame {frame_index} did not match expected annotations!"

        # If the frame is not expected to have annotations, it should be an empty list
        else:
            assert received_annotations == [], \
                f"Frame {frame_index} should have no annotations but got {received_annotations}"


def _normalize_annotation_dict(dummy_data_loader):
    normalized_expected_annotations = {
        frame_index: [
            (
                NorsvinAnnotationParser.enum_from_str(behavior),  # Convert string to enum
                x / 1920,  # Normalize X
                y / 1080,  # Normalize Y
                w / 1920,  # Normalize Width
                h / 1080  # Normalize Height
            )
            for behavior, x, y, w, h in annotations
        ]
        for frame_index, annotations in dummy_data_loader.get_annotations().items()
    }
    return normalized_expected_annotations