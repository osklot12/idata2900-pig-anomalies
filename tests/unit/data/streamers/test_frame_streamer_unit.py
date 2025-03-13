import pytest
from unittest.mock import Mock

from src.data.streamers.frame_streamer import FrameStreamer

from tests.utils.dummies.dummy_gcp_data_loader import DummyGCPDataLoader

# sample video metadata
SAMPLE_N_FRAMES = 171
SAMPLE_WIDTH = 1920
SAMPLE_HEIGHT = 1080

OUTPUT_DIR = "test_frames_output"


@pytest.fixture
def dummy_data_loader():
    """Fixture to provide a dummy GCPDataLoader."""
    return DummyGCPDataLoader(bucket_name="test-bucket", credentials_path="fake-path")


@pytest.fixture
def mock_callback():
    """Fixture to provide a mock callback."""
    return Mock()


def test_callback_called_correctly(dummy_data_loader, mock_callback):
    """Tests that the callback function is called correctly."""
    # arrange
    frame_loader = FrameStreamer(
        data_loader=dummy_data_loader,
        video_blob_name="test_video.mp4",
        callback=mock_callback,
        frame_shape=(SAMPLE_HEIGHT, SAMPLE_WIDTH),
        resize_shape=(224, 224)
    )

    # act
    frame_loader.start_streaming()
    frame_loader.wait_for_completion()

    # assert
    assert mock_callback.call_count == SAMPLE_N_FRAMES + 1, \
        f"Expected {SAMPLE_N_FRAMES + 1} calls, got {mock_callback.call_count}"

    last_call = mock_callback.call_args_list[-1]
    _, last_frame_index, last_frame, is_complete = last_call[0]
    assert last_frame is None, "Final callback did not receive None!"
    assert is_complete is True, "Final callback did not indicate completion!"


def test_no_resizing_when_resize_none(dummy_data_loader, mock_callback):
    """Tests that frames retain their original shape when resize_shape=None"""
    # arrange
    frame_loader = FrameStreamer(
        data_loader=dummy_data_loader,
        video_blob_name="test_video.mp4",
        callback=mock_callback,
        frame_shape=(SAMPLE_HEIGHT, SAMPLE_WIDTH),
        resize_shape=None
    )

    # act
    frame_loader.start_streaming()
    frame_loader.wait_for_completion()
    processed_frames = [call[0][2] for call in mock_callback.call_args_list[:-1]]

    # assert
    assert all(frame.shape == (SAMPLE_HEIGHT, SAMPLE_WIDTH, 3) for frame in processed_frames), \
        "Frames were resized when they should not have been!"


def test_resizing_works_correctly(dummy_data_loader, mock_callback):
    """Tests that frames are correctly resize when resize_shape is set."""
    # arrange
    resized_shape = (224, 224)
    frame_loader = FrameStreamer(
        data_loader=dummy_data_loader,
        video_blob_name="test_video.mp4",
        callback=mock_callback,
        frame_shape=(SAMPLE_HEIGHT, SAMPLE_WIDTH),
        resize_shape=resized_shape
    )

    # act
    frame_loader.start_streaming()
    frame_loader.wait_for_completion()
    processed_frames = [call[0][2] for call in mock_callback.call_args_list[:-1]]

    # assert
    assert all(frame.shape == (resized_shape[0], resized_shape[1], 3) for frame in processed_frames), \
        f"Some frames were not resized correctly to {resized_shape}!"


def test_correct_number_of_frames_loaded(dummy_data_loader, mock_callback):
    """Tests that the expected number of frames are loaded."""
    # arrange
    frame_loader = FrameStreamer(
        data_loader=dummy_data_loader,
        video_blob_name="test_video.mp4",
        callback=mock_callback,
        frame_shape=(SAMPLE_HEIGHT, SAMPLE_WIDTH),
        resize_shape=(224, 224)
    )

    # act
    frame_loader.start_streaming()
    frame_loader.wait_for_completion()
    processed_frame_count = len(mock_callback.call_args_list) - 1

    # assert
    assert processed_frame_count == SAMPLE_N_FRAMES, \
        f"Expected {SAMPLE_N_FRAMES} frames, got {processed_frame_count}"