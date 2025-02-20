import pytest
from unittest.mock import MagicMock, call, patch
from src.data.loading.parallel_frame_loader import ParallelFrameLoader
from src.data.loading.frame_loader import FrameLoader
from tests.utils.dummies.dummy_gcp_data_loader import DummyGCPDataLoader


@pytest.fixture
def mock_callback():
    """Fixture to provide a mock callback function."""
    return MagicMock()

@pytest.fixture
def video_list():
    """Fixture to provide a list of video names."""
    return ["video1.mp4", "video2.mp4"]

@patch("src.data.loading.frame_loader.FrameLoader.load_frames")
def test_parallel_loading(mock_load_frames, mock_callback, video_list):
    """
    Test that ParallelFrameLoader correctly spawns multiple FrameLoader instances
    and processes multiple videos in parallel.
    """

    # arrange
    mock_load_frames.return_value = None

    dummy_data_loader = DummyGCPDataLoader(bucket_name="test-bucket", credentials_path="fake-path")
    parallel_loader = ParallelFrameLoader(dummy_data_loader, mock_callback, num_threads=2)

    # act
    parallel_loader.load_frames(video_list)

    # assert
    expect_calls = [call(video) for video in video_list]
    mock_load_frames.assert_has_calls(expect_calls, any_order=True)

def test_empty_video_list(mock_callback):
    """Test handling of an empty video list."""
    # arrange
    dummy_data_loader = DummyGCPDataLoader(bucket_name="test-bucket", credentials_path="fake-path")
    parallel_loader = ParallelFrameLoader(dummy_data_loader, mock_callback)

    # act
    parallel_loader.load_frames([])

    # assert
    mock_callback.assert_not_called()

@patch("src.data.loading.frame_loader.FrameLoader.load_frames")
def test_exception_handling(mock_load_frames, mock_callback, video_list):
    """
    Test that if one FrameLoader instance fails, the others still run.
    """

    # arrange
    mock_load_frames.side_effect = [Exception("Simulated error"), None]

    dummy_data_loader = DummyGCPDataLoader(bucket_name="test-bucket", credentials_path="fake-path")
    parallel_loader = ParallelFrameLoader(dummy_data_loader, mock_callback, num_threads=2)

    # act
    parallel_loader.load_frames(video_list)

    # assert
    assert mock_load_frames.call_count == len(video_list)

@patch("src.data.loading.frame_loader.FrameLoader.load_frames")
def test_callback_is_thread_safe(mock_load_frames, mock_callback, video_list):
    """
    Tests that the callback function is correctly invoked in parallel.
    """

    # arrange
    def mock_process(video_name):
        mock_callback(video_name, 0, b"frame_data", False)
        mock_callback(video_name, 1, None, True)

    mock_load_frames.side_effect = mock_process

    dummy_data_loader = DummyGCPDataLoader(bucket_name="test-bucket", credentials_path="fake-path")
    parallel_loader = ParallelFrameLoader(dummy_data_loader, mock_callback, num_threads=2)

    # act
    parallel_loader.load_frames(video_list)

    # assert
    expected_calls = [
        call("video1.mp4", 0, b"frame_data", False),
        call("video1.mp4", 1, None, True),
        call("video2.mp4", 0, b"frame_data", False),
        call("video2.mp4", 1, None, True),
    ]
    mock_callback.assert_has_calls(expected_calls, any_order=True)