import pytest

from src.command.command import Command
from src.data.streamers.ensemble_streamer import EnsembleStreamer
from src.data.streamers.streamer_status import StreamerStatus
from tests.utils.dummies.dummy_command import DummyCommand
from tests.utils.dummies.dummy_failing_streamer import DummyFailingStreamer
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


def test_stream_and_wait_for_completion(ensemble_streamer):
    """Tests that streaming starts all streamers and they complete correctly."""
    # arrange
    ensemble_streamer.stream()

    # act
    ensemble_streamer.wait_for_completion()
    ensemble_streamer.stop()

    # assert
    for streamer_id in ensemble_streamer.get_streamer_ids():
        assert not ensemble_streamer.get_streamer(streamer_id).get_status() == StreamerStatus.STREAMING


def test_streaming_completion_status(ensemble_streamer):
    """Tests that streaming() correctly sets the status to COMPLETED if all streamers completes."""
    # arrange
    initial_status = ensemble_streamer.get_status()

    # act
    ensemble_streamer.stream()
    ensemble_streamer.wait_for_completion()
    ensemble_streamer.stop()

    end_status = ensemble_streamer.get_status()

    # assert
    assert initial_status == StreamerStatus.PENDING
    assert end_status == StreamerStatus.COMPLETED


def test_streaming_failure_status():
    """Tests that streaming() correctly sets the status to FAILED if a streamer fails."""
    # arrange
    streamer = EnsembleStreamer(
        (DummyStreamer(wait_time=.05),
         DummyFailingStreamer())
    )
    initial_status = streamer.get_status()

    # act
    streamer.stream()
    streamer.wait_for_completion()
    streamer.stop()

    end_status = streamer.get_status()

    # assert
    assert initial_status == StreamerStatus.PENDING
    assert end_status == StreamerStatus.FAILED


def test_streaming_failure_and_stopped_status():
    """Tests that streaming() correctly sets the status to FAILED if one streamer has failed while another stopped."""
    # arrange
    dummy_streamer = DummyStreamer(wait_time=.05)
    ensemble_streamer = EnsembleStreamer(
        (dummy_streamer, DummyFailingStreamer())
    )

    # act
    ensemble_streamer.stream()
    dummy_streamer.stop()
    ensemble_streamer.wait_for_completion()
    ensemble_streamer.stop()

    end_status_dummy = dummy_streamer.get_status()
    end_status_ensemble = ensemble_streamer.get_status()

    # assert
    assert end_status_dummy == StreamerStatus.STOPPED
    assert end_status_ensemble == StreamerStatus.FAILED


def test_streaming_stopped_status():
    """Tests that streaming() correctly sets the status to STOPPED if one streamer has stopped."""
    # arrange
    dummy_streamer_stop = DummyStreamer(wait_time=.05)
    dummy_streamer_complete = DummyStreamer(wait_time=.05)
    ensemble_streamer = EnsembleStreamer(
        (dummy_streamer_stop, dummy_streamer_complete)
    )

    # act
    ensemble_streamer.stream()
    dummy_streamer_stop.stop()
    ensemble_streamer.wait_for_completion()
    ensemble_streamer.stop()

    end_status_dummy_stop = dummy_streamer_stop.get_status()
    end_status_dummy_complete = dummy_streamer_complete.get_status()
    end_status_ensemble = ensemble_streamer.get_status()

    # assert
    assert end_status_dummy_stop == StreamerStatus.STOPPED
    assert end_status_dummy_complete == StreamerStatus.COMPLETED
    assert end_status_ensemble == StreamerStatus.STOPPED


def test_stop(ensemble_streamer):
    """Tests that stop() correctly sets the status to STOPPED.s"""
    # arrange
    ensemble_streamer.stream()

    # act
    ensemble_streamer.stop()
    end_status = ensemble_streamer.get_status()

    # assert
    assert end_status == StreamerStatus.STOPPED


def test_eos_commands_execution(ensemble_streamer):
    """Tests that EOS commands execute properly when all streamers complete."""
    # arrange
    eos_command = DummyCommand()

    ensemble_streamer.add_eos_command(eos_command)

    # act
    ensemble_streamer.stream()
    ensemble_streamer.wait_for_completion()
    ensemble_streamer.stop()

    # assert
    assert eos_command.executed
