import time
from unittest.mock import MagicMock

import pytest

from src.data.dataset.streams.prefetcher import Prefetcher


@pytest.fixture
def stream():
    """Fixture to provide a mock Stream instance."""
    mock = MagicMock()
    mock.read.return_value = ["dummy_batch"]
    return mock


@pytest.mark.unit
def test_prefetch_and_retrieve(stream):
    """Tests that prefetching and retrieving batches works as expected."""
    # arrange
    prefetcher = Prefetcher[str](stream=stream, buffer_size=1)

    # act
    prefetcher.run()

    time.sleep(.1)

    batch = prefetcher.read()

    # assert
    assert batch == ["dummy_batch"]
    assert stream.read.call_count >= 1

    prefetcher.stop()


@pytest.mark.unit
def test_double_run_raises(stream):
    """Tests that running the prefetcher while already running will raise."""
    # arrange
    prefetcher = Prefetcher[str](stream=stream, buffer_size=1)

    prefetcher.run()

    # act & assert
    with pytest.raises(RuntimeError, match="Prefetcher is already running"):
        prefetcher.run()

    prefetcher.stop()


@pytest.mark.unit
def test_stop_works_cleanly(stream):
    """Tests that stopping the prefetcher works as expected."""
    # arrange
    prefetcher = Prefetcher[str](stream=stream, buffer_size=1)

    prefetcher.run()
    time.sleep(.1)

    # act
    prefetcher.stop()

    # assert
    assert prefetcher._thread is None
    assert not prefetcher._running.get()


@pytest.mark.unit
def test_prefetching_halts_when_full_queue(stream):
    """Tests that prefetching temporarily halts whenever the internal queue is full."""
    # arrange
    prefetcher = Prefetcher[str](stream=stream, buffer_size=2)

    # act
    prefetcher.run()
    time.sleep(.1)

    # assert
    assert stream.read.call_count == 3

    prefetcher.stop()