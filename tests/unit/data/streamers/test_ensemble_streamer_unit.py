import pytest

from src.data.streamers.ensemble_streamer import EnsembleStreamer
from src.data.streamers.streamer_status import StreamerStatus
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
    assert len(ensemble_streamer.streamers) == len(dummy_streamers)
    assert len(ensemble_streamer.streamer_statuses) == len(dummy_streamers)

    for status in ensemble_streamer.streamer_statuses.values():
        assert status == StreamerStatus.PENDING

def test_stream_and_wait_for_completion(ensemble_streamer):
    """Tests that streaming starts all streamers and they complete correctly."""
    # arrange
    ensemble_streamer.stream()

    # act
    ensemble_streamer.wait_for_completion()

    # assert
    for streamer in ensemble_streamer.streamers.values():
        assert not streamer.streaming()