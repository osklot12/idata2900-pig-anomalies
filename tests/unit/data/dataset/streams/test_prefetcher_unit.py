import time
from unittest.mock import MagicMock

import pytest

from src.data.dataset.dataset_split import DatasetSplit
from src.data.dataset.streams.prefetcher import Prefetcher


@pytest.fixture
def batch_provider():
    """Fixture to provide a BatchProvider instance."""
    mock = MagicMock()
    mock.get_batch.return_value = ["dummy_batch"]
    return mock


@pytest.mark.unit
def test_prefetch_and_retrieve(batch_provider):
    """Tests that prefetching and retrieving batches works as expected."""
    # arrange
    prefetcher = Prefetcher[str](
        batch_provider=batch_provider,
        split=DatasetSplit.TRAIN,
        batch_size=1
    )

    # act
    prefetcher.run()

    time.sleep(.1)

    batch = prefetcher.get()

    # assert
    assert batch == ["dummy_batch"]
    assert batch_provider.get_batch.call_count >= 1

    prefetcher.stop()


@pytest.mark.unit
def test_double_run_raises(batch_provider):
    """Tests that running the prefetcher while already running will raise."""
    # arrange
    prefetcher = Prefetcher[str](
        batch_provider=batch_provider,
        split=DatasetSplit.TRAIN,
        batch_size=1
    )

    prefetcher.run()

    # act & assert
    with pytest.raises(RuntimeError, match="Prefetcher is already running"):
        prefetcher.run()

    prefetcher.stop()


@pytest.mark.unit
def test_stop_works_cleanly(batch_provider):
    """Tests that stopping the prefetcher works as expected."""
    # arrange
    prefetcher = Prefetcher[str](
        batch_provider=batch_provider,
        split=DatasetSplit.TRAIN,
        batch_size=1
    )

    prefetcher.run()
    time.sleep(.1)

    # act
    prefetcher.stop()

    # assert
    assert prefetcher._thread is None
    assert not prefetcher._running.get()


@pytest.mark.unit
def test_prefetching_halts_when_full_queue(batch_provider):
    """Tests that prefetching temporarily halts whenever the internal queue is full."""
    # arrange
    prefetcher = Prefetcher[str](
        batch_provider=batch_provider,
        split=DatasetSplit.TRAIN,
        batch_size=2,
        buffer_size=2
    )

    # act
    prefetcher.run()
    time.sleep(.1)

    # assert
    assert batch_provider.get_batch.call_count == 3

    prefetcher.stop()
