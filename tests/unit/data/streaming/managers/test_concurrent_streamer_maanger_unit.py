import pytest
from unittest.mock import MagicMock, create_autospec
import concurrent.futures

from src.data.streaming.managers.concurrent_streamer_manager import ConcurrentStreamerManager
from src.data.streaming.streamers.streamer import Streamer


class TestableConcurrentStreamerManager(ConcurrentStreamerManager):
    """A concrete implementation of ConcurrentStreamerManager for testing."""

    def _setup(self) -> None:
        self.setup_called = True

    def _run_streamer(self, streamer: Streamer, streamer_id: str) -> None:
        self.streamer_ran = streamer

    def _handle_done_streamer(self, streamer_id: str) -> None:
        self.done_called_with = streamer_id

    def _handle_crashed_streamer(self, streamer_id: str, e: Exception) -> None:
        self.crashed_called_with = (streamer_id, e)


@pytest.fixture
def manager():
    """Fixture to provide a ConcurrentStreamerManager instance."""
    return TestableConcurrentStreamerManager(max_streamers=2)


@pytest.mark.unit
def test_run_sets_running_and_calls_setup(manager):
    """Tests that running the manager sets running to True and calls _setup."""
    # act
    manager.run()

    # assert
    assert manager._running is True
    assert manager.setup_called == True


@pytest.mark.unit
def test_run_twice_raises_error(manager):
    """Tests that running the manager twice will raise an error."""
    # arrange
    manager.run()

    # act & assert
    with pytest.raises(RuntimeError, match="StreamerManager already running"):
        manager.run()


@pytest.mark.unit
def test_launch_streamer_raises_when_not_running(manager):
    """Tests that launching a streamer when not running raises an error."""
    # arrange
    mock_streamer = create_autospec(Streamer)

    # act & assert
    with pytest.raises(RuntimeError, match="Cannot launch streamer when manager is not running"):
        manager._launch_streamer(mock_streamer)


@pytest.mark.unit
def test_launch_streamer_raises_when_none(manager):
    """Tests that launching a streamer with None raises an error."""
    # arrange
    manager.run()

    # act & assert
    with pytest.raises(RuntimeError, match="Streamer cannot be None"):
        manager._launch_streamer(None)


@pytest.mark.unit
def test_launch_streamer_adds_and_starts_streamer(manager):
    """Tests that launching a streamer correctly adds and starts it."""
    # arrange
    manager.run()
    mock_streamer = create_autospec(Streamer)

    # act
    manager._launch_streamer(mock_streamer)

    # assert
    mock_streamer.start_streaming.assert_called_once()
    assert manager.streamer_ran is mock_streamer


@pytest.mark.unit
def test_on_streamer_done_calls_done_handler(manager):
    """Tests that a streamer done streaming will have handle_done_streamer called."""
    # arrange
    manager.run()
    future = concurrent.futures.Future()
    future.set_result(None)

    # act
    manager._on_streamer_done(future, "streamer1")

    # assert
    assert manager.done_called_with == "streamer1"


@pytest.mark.unit
def test_on_streamer_done_calls_crashed_handler(manager):
    """Tests that a crashing streamer will have handle_crashed_streamer called."""
    # arrange
    manager.run()
    future = concurrent.futures.Future()
    ex = RuntimeError("Oops")
    future.set_exception(ex)

    # act
    manager._on_streamer_done(future, "streamer1")

    # assert
    streamer_id, err = manager.crashed_called_with
    assert streamer_id == "streamer1"
    assert isinstance(err, RuntimeError)
    assert str(err) == "Oops"


@pytest.mark.unit
def test_stop_shuts_down_all_streamers(manager):
    """Tests that calling stop will shut down all streamers."""
    # arrange
    manager.run()
    s1 = create_autospec(Streamer)
    s2 = create_autospec(Streamer)

    manager._add_streamer(s1)
    manager._add_streamer(s2)

    # act
    manager.stop()

    # assert
    s1.stop_streaming.assert_called_once()
    s2.stop_streaming.assert_called_once()
    assert manager._running is False
    assert manager._executor is None
