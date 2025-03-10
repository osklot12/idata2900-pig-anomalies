import pytest

from src.command.command import Command
from src.data.streamers.ensemble_streamer import EnsembleStreamer
from src.data.streamers.streamer_status import StreamerStatus
from tests.utils.dummies.dummy_command import DummyCommand
from tests.utils.dummies.dummy_streamer import DummyStreamer


@pytest.fixture
def dummy_streamers():
    """Creates a tuple of two DummyStreamer instances."""
    return DummyStreamer(wait_time=.05), DummyStreamer(wait_time=.05)

@pytest.fixture
def ensemble_streamer(dummy_streamers):
    """Creates an instance of EnsembleStreamer with DummyStreamers."""
    return EnsembleStreamer(dummy_streamers)

def test_initialization(ensemble_streamer, dummy_streamers):
    """Tests that the EnsembleStreamer initializes correctly."""
    # assert
    assert len(ensemble_streamer._streamers) == len(dummy_streamers)
    assert len(ensemble_streamer._streamer_statuses) == len(dummy_streamers)

    for status in ensemble_streamer._streamer_statuses.values():
        assert status == StreamerStatus.PENDING

def test_stream_and_wait_for_completion(ensemble_streamer):
    """Tests that streaming starts all streamers and they complete correctly."""
    # arrange
    ensemble_streamer.stream()

    # act
    ensemble_streamer.wait_for_completion()

    # assert
    for streamer in ensemble_streamer._streamers.values():
        assert not streamer.streaming()

def test_streaming_status(ensemble_streamer):
    """Tests that streaming() correctly reflects state."""
    # arrange & act
    first_streaming_check = ensemble_streamer.streaming()
    ensemble_streamer.stream()

    second_streaming_check = ensemble_streamer.streaming()
    ensemble_streamer.wait_for_completion()

    third_streaming_check = ensemble_streamer.streaming()

    # assert
    assert not first_streaming_check
    assert second_streaming_check
    assert not third_streaming_check

def test_stop(ensemble_streamer):
    """Tests that stopping the EnsembleStreamer stops all streamers."""
    # arrange
    ensemble_streamer.stream()

    # act
    ensemble_streamer.stop()

    # assert
    streamer_ids = ensemble_streamer._streamers.keys()
    for streamer_id in streamer_ids:
        assert not ensemble_streamer._streamers.get(streamer_id).streaming()

def test_eos_commands_execution(ensemble_streamer):
    """Tests that EOS commands execute properly when all streamers complete."""
    # arrange
    eos_command = DummyCommand()

    ensemble_streamer.add_eos_command(eos_command)

    # act
    ensemble_streamer.stream()
    ensemble_streamer.wait_for_completion()

    # assert
    assert eos_command.executed