import pytest

from src.data.streaming.streamers import StreamerStatus
from tests.utils.dummies.dummy_command import DummyCommand
from tests.utils.dummies.dummy_failing_streamer import DummyFailingStreamer
from tests.utils.dummies.dummy_streamer import DummyStreamer


@pytest.fixture
def streamer():
    """Fixture to create a DummyStreamer instance."""
    return DummyStreamer(wait_time=.05)


@pytest.fixture
def failing_streamer():
    """Fixture to create a DummyFailingStreamer instance."""
    return DummyFailingStreamer()


def test_initial_status(streamer):
    """Tests that a newly created streamer is in PENDING state."""
    assert streamer.get_status() == StreamerStatus.PENDING


def test_stream_starts_correctly(streamer):
    """Tests that streaming updates the status and starts a thread."""
    streamer.start_streaming()
    assert streamer.get_status() == StreamerStatus.STREAMING


def test_stream_completes_correctly(streamer):
    """Tests that a streamer completes and updates its status."""
    streamer.start_streaming()
    streamer.wait_for_completion()
    assert streamer.get_status() == StreamerStatus.COMPLETED


def test_wait_for_completion_when_not_streaming(streamer):
    """Tests that calling wait_for_completion() when not streaming should not raise an exception."""
    # act
    streamer.wait_for_completion()


def test_stop(streamer):
    """Tests that stopping a streamer updates its status to STOPPED."""
    # arrange
    streamer.start_streaming()

    # act
    streamer.stop()

    # assert
    assert streamer.get_status() == StreamerStatus.STOPPED


def test_eos_command_executes_correctly(streamer):
    """Tests that eos commands executes correctly."""
    # arrange
    cmd_one = DummyCommand()
    cmd_two = DummyCommand()
    streamer.add_eos_command(cmd_one)
    streamer.add_eos_command(cmd_two)

    # act
    streamer.start_streaming()
    streamer.wait_for_completion()

    # assert
    assert cmd_one.executed
    assert cmd_two.executed


def test_status_set_on_failure(failing_streamer):
    """Tests that a streamer raising an exception while streaming updates the status to FAILED."""
    # act
    failing_streamer.start_streaming()
    failing_streamer.wait_for_completion()

    # assert
    assert failing_streamer.get_status() == StreamerStatus.FAILED


def test_eos_command_executes_on_failure(failing_streamer):
    """Tests that eos commands executes when an exception is raised while streaming."""
    # arrange
    cmd_one = DummyCommand()
    cmd_two = DummyCommand()

    failing_streamer.add_eos_command(cmd_one)
    failing_streamer.add_eos_command(cmd_two)

    # act
    failing_streamer.start_streaming()
    failing_streamer.wait_for_completion()

    assert cmd_one.executed
    assert cmd_two.executed
