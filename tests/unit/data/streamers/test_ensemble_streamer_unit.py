from unittest.mock import MagicMock

import pytest
import time

from src.command.command import Command
from src.data.streamers.ensemble_streamer import EnsembleStreamer
from tests.utils.dummies.dummy_streamer import DummyStreamer


@pytest.fixture
def mock_streamers():
    """Fixture to create multiple mock streamers."""
    return [DummyStreamer() for _ in range(5)]


@pytest.fixture
def mock_termination_command():
    """Fixture to create a mock termination command."""
    return MagicMock(spec=Command)


def test_stream_starts_all_streamers(mock_streamers, mock_termination_command):
    """Tests that `stream()` starts all streamers."""
    # arrange
    ensemble = EnsembleStreamer(streamers=tuple(mock_streamers))

    # act
    ensemble.stream()

    # assert
    assert all(streamer.running for streamer in mock_streamers)

    ensemble.stop()


def test_stop_stops_all_streamers(mock_streamers):
    """Tests that `stop()` stops all streamers."""
    # arrange
    ensemble = EnsembleStreamer(tuple(mock_streamers))
    ensemble.stream()

    # act
    ensemble.stop()

    # assert
    assert all(not streamer.running for streamer in mock_streamers)


def test_wait_for_completion_block_until_done(mock_streamers):
    """Tests that `wait_for_completion()` blocks until streaming is complete."""
    # arrange
    ensemble = EnsembleStreamer(tuple(mock_streamers))
    start_time = time.time()
    ensemble.stream()

    # act
    ensemble.wait_for_completion()

    # assert
    duration = time.time() - start_time
    assert duration >= (mock_streamers[0].wait_time * len(mock_streamers))


def test_stream_runs_in_background(mock_streamers, mock_termination_command):
    """Tests that `stream_release_notify()` runs asynchronously."""
    # arrange
    ensemble = EnsembleStreamer(streamers=tuple(mock_streamers), termination_command=mock_termination_command)

    # act
    ensemble.stream()
    ensemble.wait_for_completion()

    # assert
    mock_termination_command.execute.assert_called_once()


def test_stop_can_be_called_multiple_times(mock_streamers):
    """Tests that `stop()` can be called multiple times safely."""
    # arrange
    ensemble = EnsembleStreamer(tuple(mock_streamers))
    ensemble.stream()

    # act
    ensemble.stop()
    ensemble.stop()

    # assert
    assert all(not streamer.running for streamer in mock_streamers)


def test_cannot_stream_twice_without_stopping(mock_streamers):
    """Tests that `stream()` cannot be called twice in a row without stopping first."""
    # arrange
    ensemble = EnsembleStreamer(tuple(mock_streamers))
    ensemble.stream()

    # act & assert
    with pytest.raises(RuntimeError, match="Streamers already running on another thread."):
        ensemble.stream()


def test_stop_does_not_fail_if_never_started(mock_streamers):
    """Tests that calling `stop()` without starting does not fail."""
    # arrange
    ensemble = EnsembleStreamer(tuple(mock_streamers))

    # act & assert
    ensemble.stop()

def test_wait_for_completion_does_not_fail_if_never_started(mock_streamers):
    """Tests that calling `wait_for_completion()` without starting does not fail."""
    # arrange
    ensemble = EnsembleStreamer(tuple(mock_streamers))

    # act & assert
    ensemble.wait_for_completion()