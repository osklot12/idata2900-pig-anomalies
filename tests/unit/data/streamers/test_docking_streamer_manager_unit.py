import pytest
import time

from src.data.streamers.docking_streamer_manager import DockingStreamerManager
from tests.utils.dummies.dummy_streamer_provider import DummyStreamerProvider


@pytest.fixture
def manager():
    """Fixture to provide a DockingStreamerManager instance."""
    return DockingStreamerManager(DummyStreamerProvider(), 4)

def test_initialization_with_valid_arguments():
    """Tests that initialization works correctly."""
    # arrange
    n_streamers = 2
    provider = DummyStreamerProvider()

    # act
    manager = DockingStreamerManager(provider, n_streamers)

    # assert
    assert manager._streamer_provider == provider
    assert manager._n_streamers == n_streamers
    assert manager.n_active_streamers() == 0


def test_initialization_with_invalid_n_streamers():
    """Tests that initialization with an invalid number of streamers fails."""
    # arrange
    n_streamers = -1
    provider = DummyStreamerProvider()

    # act
    with pytest.raises(ValueError) as exc_info:
        DockingStreamerManager(provider, n_streamers)

    # assert
    assert str(exc_info.value) == "n_streamers must be greater than 0"


def test_run_should_activate_n_streamers(manager):
    """Tests that running the manager immediately activates n streamers."""
    # act
    manager.run()
    n_streamers = manager.n_active_streamers()
    manager.stop()

    # assert
    assert n_streamers == 4

def test_streamers_are_replaced_after_completion(manager):
    """Tests that streamers are replaced after the complete."""
    # arrange
    provider = DummyStreamerProvider(n_streamers=5, streamer_wait_time=.1)
    manager = DockingStreamerManager(provider, 2)

    # act
    manager.run()
    time.sleep(0.15)
    active_streamers = manager.n_active_streamers()
    manager.stop()

    # assert
    assert active_streamers == 2

def test_no_more_streamers_should_be_handled_correctly():
    """Tests that running of out streamers in StreamerProvider is handled gracefully."""
    # arrange
    provider = DummyStreamerProvider(n_streamers=5, streamer_wait_time=.1)
    manager = DockingStreamerManager(provider, 4)

    # act
    manager.run()
    time.sleep(0.15)
    active_streamers = manager.n_active_streamers()
    manager.stop()

    # assert
    assert active_streamers == 1

def test_stop_should_prevent_new_streamers():
    """Tests that stopping the manager prevents new streamers from starting."""
    # arrange
    provider = DummyStreamerProvider(n_streamers=10, streamer_wait_time=.2)
    manager = DockingStreamerManager(provider, 3)

    # act
    manager.run()
    time.sleep(.5)
    manager.stop()
    active_after_stop = manager.n_active_streamers()

    # assert
    assert active_after_stop == 0

