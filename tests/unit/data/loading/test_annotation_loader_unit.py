from unittest.mock import Mock

import pytest

from src.data.decoders.darwin_decoder import DarwinDecoder
from src.data.loading.annotation_loader import AnnotationLoader
from src.data.bbox_normalizer import BBoxNormalizer
from src.utils.norsvin_behavior_class import NorsvinBehaviorClass
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
    # Arrange
    annotation_loader = AnnotationLoader(
        data_loader=dummy_data_loader,
        decoder_cls=DarwinDecoder,
        label_parser=NorsvinBehaviorClass,
        callback=mock_callback
    )

    # Act
    annotation_loader.load_annotations("test_annotations.json")
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
    annotation_loader = AnnotationLoader(
        data_loader=dummy_data_loader,
        decoder_cls=DarwinDecoder,
        callback=mock_callback
    )

    # act
    annotation_loader.load_annotations("test_annotations.json")
    annotation_loader.wait_for_completion()

    raw_expected_annotations = dummy_data_loader.get_annotations()

    expected_annotations = {}
    original_width, original_height = dummy_data_loader.get_video_properties()[:2]
    new_range = (0, 1)  # The expected normalization range

    for frame, annotations in raw_expected_annotations.items():
        normalized_annotations = [
            (
                NorsvinBehaviorClass.from_json_label(behavior),
                *BBoxNormalizer.normalize_bounding_box(
                    image_dimensions=(original_width, original_height),
                    bounding_box=(x, y, w, h),
                    new_range=new_range,
                )
            )
            for behavior, x, y, w, h in annotations
        ]
        expected_annotations[frame] = normalized_annotations

    # Assert
    total_frames = dummy_data_loader.frame_count  # Use stored frame count

    for frame_index in range(total_frames):
        call = mock_callback.call_args_list[frame_index]
        _, received_frame, received_annotations, is_complete = call[0]

        # If the final callback, ensure it's signaling completion
        if is_complete:
            assert received_frame is None, "Final callback should not have a frame index!"
            assert received_annotations is None, "Final callback should not have annotations!"
            continue

        # If the frame is expected to have annotations, check if they match
        if frame_index in expected_annotations:
            assert received_annotations == expected_annotations[frame_index], \
                f"Annotations at frame {frame_index} did not match expected annotations!"

        # If the frame is not expected to have annotations, it should be an empty list
        else:
            assert received_annotations == [], \
                f"Frame {frame_index} should have no annotations but got {received_annotations}"